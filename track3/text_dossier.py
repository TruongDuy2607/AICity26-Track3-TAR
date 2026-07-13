"""Collision-Dossier text generation for the 4 BERTScore paragraph tasks.

ISOLATED, REMOVABLE add-on (like ``temporal_grounding`` / the retired RFT): a
single module + ``scripts/text_dossier.sh``. It touches NO existing decode path —
it only *produces* a ``{item_index, task, prediction}`` jsonl that a NEW
``track3.structural`` rule ``--text-override`` splices over the model's text
predictions. Removal = ``git rm track3/text_dossier.py scripts/text_dossier.sh``
+ revert the ``apply_text_override`` hook in ``structural.py``.

----------------------------------------------------------------------------
WHY (the leaderboard decomposition — see the analysis this followed)
----------------------------------------------------------------------------
Temporal is solved (the per-frame grounding put mIoU at rank-1 level). After that,
**100 % of the gap to rank-1 is the four free-text BERTScore tasks that narrate
collision dynamics**: ``temporal_description``, ``causal_linkage``,
``video_summarization``, ``open_qa``. Rank-1 proves it: they score WORSE than us on
BCQ/MCQ (no structural pairing) yet rank #1 purely on these four columns. Matching
their text level while keeping our structural wins lands ≈0.62 — a decisive #1.

The four tasks are not a register problem (curating phrasing was tried and did not
move them); the bottleneck is CONTENT — accurately narrating *which vehicle hit
which, in what order, why, and the result* — which our 8B model trained mostly on
normal-flow surveillance does poorly.

----------------------------------------------------------------------------
KEY STRUCTURAL INSIGHT — the text tasks leak content to each other
----------------------------------------------------------------------------
On every clip ALL text tasks describe THE SAME single collision, and the OTHER
tasks' questions / our own high-accuracy answers hand us the facts for free:

  * ``temporal_localization`` question NAMES the event as a clean referring
    expression: "When does *the T-bone collision between the black SUV and the
    black sedan* occur?".
  * ``mcq`` question + our chosen option (MCQ accuracy ~0.80+) gives the
    consequence: "...direct consequence... -> Both vehicles are disabled and stall
    in the middle of the intersection".
  * ``temporal_description`` / ``causal_linkage`` questions carry the collision
    window ``[X, Y]`` (the same timestamps as the temporal leak).
  * ``bcq_openended`` answers carry one-sentence verified observations.

So: assemble a per-clip FACT SHEET from the released questions + our predictions,
make ONE grounded "dossier" pass over the video (the single perception step), then
RENDER each task answer from the dossier in that task's register + length. One good
dossier lifts all four columns at once, and every fact is sourced from the released
test questions or our own outputs (FAQ-clean — no test annotation).

----------------------------------------------------------------------------
ALGORITHM (per clip)
----------------------------------------------------------------------------
1. ``clip_facts`` — event phrase (tl question), window ``[X,Y]`` (td/cl question),
   consequence (mcq question + our chosen letter), observations (bcq_oe answers).
2. ONE dossier generation over the video, seeded with the fact sheet, emitting a
   labelled SCENE / VEHICLES / SEQUENCE / CAUSE / AFTERMATH account with timestamps.
3. Per target task, RENDER the answer from (dossier [+ video] + verbatim question)
   in the task's concise factual register.
4. ``calibrate_length`` (trim to the per-task train-median budget; never pad) +
   the task's own ``submission`` enforcement (guaranteed non-empty, no <think>).

Output jsonl line: ``{item_index, video_id, task, prediction, source}``. Items NOT
covered fall back to the model's own text (the override is partial-safe), so this
degrades gracefully and is gated on curated-val before any submission.

NOTE (v5-2 register lesson): feeding the model an analysis + question is OUT of the
SFT distribution and rescaled BERTScore is sensitive to register drift — so this
NEVER ships without a positive delta on the curated-val gate (postprocess.sh with
GT_JSON=val_gt_curated.json). Defaults keep the video attached during the render
(``render_with_video``) to stay as close to the SFT input as possible.

Heavy deps (ms-swift, torch) are imported lazily so the pure assembly logic
(facts / prompts / length) is unit-testable on a CPU dev box with a stub generator.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field

from track3.evidence import (
    EvidenceSheet,
    classify_stem,
    extract_cast,
    jitter_evidence,
    render_evidence_prompt,
)
from track3.structural import (
    load_items_by_video,
    normalize_ts,
    question_options,
    question_window,
    resolve_video,
)
from track3.tasks import extract_letter, extract_yesno, get_task, strip_reasoning
from track3.temporal_grounding import parse_event_phrase

# Default target set: the 4 long-narrative BERTScore tasks (proven +0.0268 mean).
TARGET_TASKS = (
    "temporal_description",
    "causal_linkage",
    "open_qa",
    "video_summarization",
)

# P1 extension (opt-in via --tasks): the remaining BERTScore text tasks the dossier
# can also serve — they are content gaps too (measured: SD .381 vs .410, MCQ-OE .777
# vs .831, BCQ-OE .582 vs .606). scene_description is the dossier's SCENE part; the
# *_openended twins reuse the dossier for the EXPLANATION while the leading token
# (letter / Yes-No) is kept from the model/structural prediction. Gate on curated-val.
EXTENDED_TASKS = TARGET_TASKS + ("scene_description", "mcq_openended", "bcq_openended")

# Tasks whose answer is "<token>. <explanation>": render the explanation from the
# dossier, then restore the reliable leading token. In the full pipeline structural
# rules 4.2/4.3 refine the token again (they run AFTER --text-override).
_CLOSED_TWIN = frozenset({"mcq_openended", "bcq_openended"})

# Per-task train-GT median answer length in characters (data.md §2) — the length
# budget the render is trimmed toward (never padded). BERTScore-rescaled penalizes
# gross length mismatch; the budget caps runaway generations without cutting content.
LENGTH_TARGET = {
    "temporal_description": 394,
    "causal_linkage": 657,
    "open_qa": 426,
    "video_summarization": 617,
    "scene_description": 645,
    "mcq_openended": 98,    # "X. <one-sentence reason>"
    "bcq_openended": 97,    # "Yes./No. <one-sentence reason>"
}

# causal_linkage / temporal_description answers reference the question's window in
# the train register; rule 4.4 (timestamp_echo) is the post-hoc safety net but we
# also ask the render to include them so the echo is natural, not bolted on.
_WINDOW_TASKS = frozenset({"causal_linkage", "temporal_description"})


# ---------------------------------------------------------------------------
# Pure assembly logic — no ms-swift / torch (unit-testable on the dev box)
# ---------------------------------------------------------------------------

# The per-clip fact container IS the canonical Evidence Sheet v2 (track3.evidence)
# — one schema shared verbatim with the SFT dataset builder, so the direct-mode
# render prompt can never diverge from what the model was trained on.
ClipFacts = EvidenceSheet


def _mcq_facts(tasks: dict, preds: dict) -> tuple[str, str]:
    """(cause, consequence) = the chosen option TEXTS of BOTH mcq-family items —
    our high-accuracy answers re-used as verified facts. Root-cause/fault stems
    feed 'cause' (evidence.classify_stem), the rest 'consequence'; duplicate
    option texts (the shuffled twins, data.md §5.3) dedupe to one line."""
    cause = consequence = ""
    seen: set[str] = set()
    for src in ("mcq", "mcq_openended"):
        for it in tasks.get(src, ()):
            rec = preds.get(str(it["item_index"]))
            letter = extract_letter((rec or {}).get("prediction") or "") or ""
            opts = question_options(it["question"], normalize=False)
            if letter not in opts:
                continue
            key = re.sub(r"\W+", " ", opts[letter]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            if classify_stem(it["question"]) == "cause" and not cause:
                cause = opts[letter]
            elif not consequence:
                consequence = opts[letter]
    return cause, consequence


def _observations(tasks: dict, preds: dict, limit: int = 2) -> list[str]:
    """One-sentence verified observations from the bcq_openended answers."""
    out = []
    for it in tasks.get("bcq_openended", ()):
        rec = preds.get(str(it["item_index"]))
        text = (rec or {}).get("prediction") or ""
        text = text.strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def clip_facts(tasks: dict, preds: dict | None = None,
               scene: tuple[str, ...] = ()) -> EvidenceSheet:
    """Assemble the per-clip Evidence Sheet v2 from the released questions + our
    own predictions (PROVE.md §1 tier 1).

    ``tasks`` is one video's ``{task_type: [items]}`` (from
    ``structural.load_items_by_video``). ``preds`` (item_index -> record) is
    optional — without it only the question-derived facts (event, window, cast)
    are available, which is enough to run and is what the unit tests exercise.
    ``scene`` is the clip's W1-probed attribute list (claim_verify --scene-out).
    """
    preds = preds or {}
    video_id = ""
    event = "the collision or anomaly"
    for it in tasks.get("temporal_localization", ()):
        video_id = it.get("video_id", video_id)
        event = parse_event_phrase(it.get("question", "")) or event
        break
    if not video_id:  # no tl item — take any item's video_id
        for its in tasks.values():
            if its:
                video_id = its[0].get("video_id", "")
                break

    window = None
    for src in ("temporal_description", "causal_linkage"):
        for it in tasks.get(src, ()):
            w = question_window(it.get("question", ""))
            if w:
                window = (normalize_ts(w[0]), normalize_ts(w[1]))
                break
        if window:
            break

    cause, consequence = _mcq_facts(tasks, preds)
    return EvidenceSheet(
        video_id=video_id,
        event_phrase=event,
        window=window,
        cast=extract_cast(q.get("question", "") for t in sorted(tasks)
                          for q in tasks[t]),
        scene=tuple(scene or ()),
        cause=cause,
        consequence=consequence,
        observations=_observations(tasks, preds),
    )


# --- prompts ---------------------------------------------------------------

SYS_DOSSIER = (
    "You are an expert traffic-accident analyst. You are shown a short traffic "
    "video. Produce a single, factual, grounded analysis of the incident using "
    "ONLY what is visible in the footage. Do not speculate beyond what is shown."
)

SYS_RENDER = (
    "You are an expert traffic-surveillance video analyst. Answer the question "
    "using the verified analysis (and the video, if shown). Be concise, factual, "
    "and specific, in the style of the analysis; do not add unverified details."
)


def dossier_prompt(facts: EvidenceSheet) -> str:
    """User turn for the single grounded perception pass, seeded with the facts."""
    lines = ["<video>", "Verified facts about this clip (use them; do not contradict them):",
             f"- Key event: {facts.event_phrase}."]
    if facts.window:
        lines.append(f"- It occurs between {facts.window[0]} and {facts.window[1]}.")
    if facts.cast:
        lines.append("- Vehicles involved: " + "; ".join(facts.cast) + ".")
    if facts.scene:
        lines.append("- Scene: " + ", ".join(facts.scene) + ".")
    if facts.cause:
        lines.append(f"- Root cause: {facts.cause}")
    if facts.consequence:
        lines.append(f"- Direct consequence: {facts.consequence}")
    for obs in facts.observations:
        lines.append(f"- Observation: {obs}")
    lines += [
        "",
        "Write a detailed, factual account of the incident with these labelled parts:",
        "SCENE: road layout, setting, time of day, weather/visibility.",
        "VEHICLES: the vehicles/agents involved and how they move.",
        "SEQUENCE: what happens step by step, with timestamps (mm:ss).",
        "CAUSE: why the incident happens.",
        "AFTERMATH: the immediate result and consequences.",
    ]
    return "\n".join(lines)


def render_prompt(task: str, question: str, dossier: str,
                  facts: ClipFacts, with_video: bool) -> str:
    """User turn that renders one task's answer from the dossier + question."""
    lines = []
    if with_video:
        lines.append("<video>")
    lines += ["Verified analysis of the video:", dossier.strip(), "",
              question.strip()]
    if task in _WINDOW_TASKS and facts.window:
        lines.append(f"Refer to the moments {facts.window[0]} and {facts.window[1]} "
                     "explicitly in your answer.")
    return "\n".join(lines)


_SENT = re.compile(r"(?<=[.!?])\s+")
_LEAD = re.compile(r"^\s*(?:yes|no|[A-D])\b[.,:)]?\s*", re.IGNORECASE)


def _twin_token(task: str, item: dict, preds: dict) -> str:
    """Reliable leading token for a *_openended twin, taken from the MODEL/structural
    prediction (never the dossier): mcq_openended -> a letter, bcq_openended -> Yes/No.
    Falls back to a safe default when the prediction is missing/unparseable."""
    p = (preds.get(str(item["item_index"])) or {}).get("prediction") or ""
    if task == "mcq_openended":
        return extract_letter(p) or "A"
    return (extract_yesno(p) or "no").capitalize()


def _prefix_token(text: str, token: str) -> str:
    """Replace any leading Yes/No/letter in ``text`` with ``token`` (keep the rest)."""
    rest = _LEAD.sub("", (text or "").strip(), count=1)
    return f"{token}. {rest}".strip()


def calibrate_length(text: str, budget: int) -> str:
    """Trim ``text`` to <= ``budget`` chars on a sentence boundary; never pad.

    Keeps whole sentences until the next would overflow the budget. If even the
    first sentence overflows, hard-cut (so a runaway generation can't blow up the
    submission). ``budget <= 0`` disables trimming.
    """
    text = (text or "").strip()
    if budget <= 0 or len(text) <= budget:
        return text
    out = ""
    for s in _SENT.split(text):
        cand = f"{out} {s}".strip() if out else s
        if len(cand) > budget:
            if not out:                      # first sentence alone overflows
                return cand[:budget].rstrip()  # hard cut (never pad)
            break
        out = cand
    return out


# ---------------------------------------------------------------------------
# IO boundary — the video-grounded generator (lazy heavy deps; injectable)
# ---------------------------------------------------------------------------

class DossierGenerator:
    """Batched text generation over the ms-swift engine (reuses infer.build_engine
    so adapter / vLLM / transformers handling never diverges).

    ``__call__(jobs)`` takes ``[{system, user, video}]`` (``video`` is a path or
    None) and returns one generated string per job (``<think>`` stripped). Native
    video is attached when ``video`` is set, matching the SFT input distribution.
    """

    def __init__(self, args, max_new_tokens: int = 512, temperature: float = 0.0,
                 batch_size: int = 256):
        from swift import InferRequest, RequestConfig  # lazy
        from track3.infer import build_engine
        self._InferRequest = InferRequest
        self._cfg = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        self.engine, adapters = build_engine(args)
        self.extra = {}
        if adapters and args.backend == "vllm":
            from swift import AdapterRequest
            self.extra = {"adapter_request": AdapterRequest("track3", adapters[0])}
        self.batch_size = max(1, batch_size)

    def _build(self, j: dict):
        kw = {"messages": [{"role": "system", "content": j["system"]},
                           {"role": "user", "content": j["user"]}]}
        if j.get("video"):
            kw["videos"] = [j["video"]]
        return self._InferRequest(**kw)

    def __call__(self, jobs: list[dict]) -> list[str]:
        """Chunked batched generation, resilient to a single bad video.

        Requests go to ``engine.infer`` in chunks of ``batch_size``: ms-swift
        preprocesses a batch up-front and every video-attached request decord-
        decodes its clip into host RAM at encode time, so one giant batch (e.g.
        the K-sample candidate pool, ~4k requests) OOM-kills the process. Within
        a failed chunk we retry per-request so one corrupt clip only loses its
        own item (returns "" -> the caller leaves it to the model's own text).
        Missing files are pre-filtered in ``run`` (cheaper).
        """
        out: list[str] = []
        for lo in range(0, len(jobs), self.batch_size):
            reqs = [self._build(j) for j in jobs[lo:lo + self.batch_size]]
            try:
                resp = self.engine.infer(reqs, self._cfg, **self.extra)
                out.extend(strip_reasoning(r.choices[0].message.content) for r in resp)
            except Exception as e:
                print(f"[text_dossier][WARN] chunk infer failed ({e}); retrying "
                      f"{len(reqs)} request(s) individually.")
                for r in reqs:
                    try:
                        resp = self.engine.infer([r], self._cfg, **self.extra)
                        out.append(strip_reasoning(resp[0].choices[0].message.content))
                    except Exception as e2:
                        print(f"[text_dossier][WARN] request failed, leaving to model: {e2}")
                        out.append("")
            if len(jobs) > self.batch_size:
                print(f"[text_dossier] generated {min(lo + self.batch_size, len(jobs))}"
                      f"/{len(jobs)}", flush=True)
        return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class DossierConfig:
    tasks: tuple[str, ...] = TARGET_TASKS
    render_with_video: bool = True   # keep video in render (closest to SFT input)
    length_tol: float = 0.3          # budget = train-median * (1 + tol); <0 disables
    dossier_max_tokens: int = 512
    render_max_tokens: int = 400
    temperature: float = 0.0
    # DIRECT mode: skip the 2-pass analysis and render straight from the evidence
    # sheet (evidence.render_evidence_prompt) — the interface the render-SFT-distilled
    # model was trained on. Default off -> the classic 2-pass dossier is unchanged.
    direct: bool = False
    fact_policy: str = "video-priority"
    # PROVE tier 3 — K-candidate sampling for the MBR ensemble (mbr_select.py):
    # candidate 0 is the greedy full-evidence render; candidates 1..K-1 are sampled
    # (sample_temperature) under fact-subset jitter (direct mode). 1 = off.
    n_samples: int = 1
    sample_temperature: float = 0.8
    candidates_out: str = ""         # write {item_index, task, candidates} jsonl here
    seed: int = 0                    # jitter determinism


def _budget(task: str, cfg: DossierConfig) -> int:
    if cfg.length_tol < 0:
        return 0
    return round(LENGTH_TARGET.get(task, 0) * (1.0 + cfg.length_tol))


def _finalize(text: str, task: str, item: dict, preds: dict,
              cfg: DossierConfig) -> str:
    """Length-calibrate + restore the closed-twin token + format-enforce ONE render;
    '' when the render was empty (the caller falls back to the model's own text)."""
    if not (text or "").strip():
        return ""
    text = calibrate_length(text, _budget(task, cfg))
    if task in _CLOSED_TWIN:  # dossier supplies the reason; keep the closed token
        text = _prefix_token(text, _twin_token(task, item, preds))
    return get_task(task).submission(text)  # enforce non-empty, strip tags


def run(items_by_video: dict, preds: dict, videos_root: str, out_path: str,
        cfg: DossierConfig, dossier_gen, render_gen=None, sample_gen=None,
        scene_map: dict | None = None, exists_fn=os.path.exists) -> list[dict]:
    """Produce one override record per (clip, target task) item.

    ``dossier_gen(jobs)`` runs the per-clip video pass; ``render_gen`` (defaults to
    ``dossier_gen``) runs the greedy per-task render; ``sample_gen`` (defaults to
    ``render_gen``) runs the temperature-sampled candidate renders when
    ``cfg.n_samples > 1``. All are injected (the real :class:`DossierGenerator`
    above; stubs in the tests) and each family is generated in ONE batched call.
    ``scene_map`` ({video_id: [scene attributes]}, from ``claim_verify
    --scene-out``) feeds the evidence sheet's Scene line.

    Clips whose video file is absent (``exists_fn``) are skipped — their items
    fall back to the model's own text (the override is partial-safe). Tests inject
    ``exists_fn=lambda _: True`` to use fake paths.
    """
    render_gen = render_gen or dossier_gen
    scene_map = scene_map or {}
    target = set(cfg.tasks)

    # 1) one clip per video with a target item AND a video (facts always built).
    clips, missing = [], 0
    for vid, tasks in items_by_video.items():
        if not any(tasks.get(t) for t in target):
            continue
        vpath = resolve_video(videos_root, vid)
        if not exists_fn(vpath):
            missing += 1
            continue
        facts = clip_facts(tasks, preds, scene=tuple(scene_map.get(vid, ())))
        clips.append({"vid": vid, "tasks": tasks, "facts": facts, "vpath": vpath})
    if missing:
        print(f"[text_dossier][WARN] {missing} clip(s) have no video under "
              f"{videos_root} — skipped (those items keep the model's text). "
              "For a val split set GROUND_VIDEOS_ROOT=$TRAIN_VIDEOS_ROOT.")

    if cfg.direct:
        # DIRECT: no analysis pass — render straight from the evidence sheet (parity
        # with the render-SFT-distilled model's training interface).
        dossiers = [None] * len(clips)
    else:
        # 1b) one analysis job per clip, then render conditioned on the analysis.
        dossier_jobs = [{"system": SYS_DOSSIER, "user": dossier_prompt(c["facts"]),
                         "video": c["vpath"]} for c in clips]
        dossiers = dossier_gen(dossier_jobs) if dossier_jobs else []
        if len(dossiers) != len(dossier_jobs):  # backend failure -> no override at all
            print(f"[text_dossier][WARN] generator returned {len(dossiers)} dossiers "
                  f"for {len(dossier_jobs)} clips; emitting no override (falls back to "
                  "the model's own text).")
            dossiers = [""] * len(dossier_jobs)

    # 2) one greedy render job per (clip, target task) item.
    plans, render_jobs = [], []
    for clip, dossier in zip(clips, dossiers):
        if not cfg.direct and not dossier:
            continue  # no dossier for this clip -> leave its items to the model
        for task in cfg.tasks:
            for it in clip["tasks"].get(task, ()):
                question = it.get("question", "")
                if cfg.direct:
                    user = render_evidence_prompt(task, question, clip["facts"],
                                                  cfg.render_with_video, cfg.fact_policy)
                else:
                    user = render_prompt(task, question, dossier,
                                         clip["facts"], cfg.render_with_video)
                video = clip["vpath"] if cfg.render_with_video else None
                render_jobs.append({"system": SYS_RENDER, "user": user, "video": video})
                plans.append({"item": it, "task": task, "question": question,
                              "facts": clip["facts"], "dossier": dossier,
                              "video": video})

    renders = render_gen(render_jobs) if render_jobs else []
    if len(renders) != len(render_jobs):
        print(f"[text_dossier][WARN] generator returned {len(renders)} renders for "
              f"{len(render_jobs)} items; emitting no override.")
        return _write([], out_path)

    # 2b) PROVE tier 3 — K-1 sampled candidates per item. Direct mode additionally
    #     jitters the evidence sheet per candidate (one optional fact dropped —
    #     mirrors the training noise, decorrelates the MBR pool). A sampling
    #     failure only loses the extra candidates, never the greedy override.
    samples_by_plan: list[list[str]] = [[] for _ in plans]
    if cfg.n_samples > 1 and plans:
        sgen = sample_gen or render_gen
        sample_jobs, owners = [], []
        for pi, pl in enumerate(plans):
            for k in range(1, cfg.n_samples):
                rng = random.Random(f"{cfg.seed}:{pl['item'].get('item_index')}:{k}")
                if cfg.direct:
                    sheet = jitter_evidence(pl["facts"], rng)
                    user = render_evidence_prompt(pl["task"], pl["question"], sheet,
                                                  cfg.render_with_video, cfg.fact_policy)
                else:
                    user = render_prompt(pl["task"], pl["question"], pl["dossier"],
                                         pl["facts"], cfg.render_with_video)
                sample_jobs.append({"system": SYS_RENDER, "user": user,
                                    "video": pl["video"]})
                owners.append(pi)
        sampled = sgen(sample_jobs) if sample_jobs else []
        if len(sampled) == len(sample_jobs):
            for pi, text in zip(owners, sampled):
                samples_by_plan[pi].append(text)
        else:
            print(f"[text_dossier][WARN] sample generator returned {len(sampled)} of "
                  f"{len(sample_jobs)} candidates — greedy-only (no MBR pool).")

    # 3) finalize every candidate identically (MBR must choose among
    #    submission-ready texts); collect the greedy override + the candidate pool.
    records, cand_records, failed = [], [], 0
    for pi, (pl, text) in enumerate(zip(plans, renders)):
        task = pl["task"]
        final = _finalize(text, task, pl["item"], preds, cfg)
        cands = [final] + [_finalize(s, task, pl["item"], preds, cfg)
                           for s in samples_by_plan[pi]]
        cands = [c for c in cands if c]
        if cfg.candidates_out and cands:
            cand_records.append({"item_index": pl["item"]["item_index"],
                                 "video_id": pl["item"].get("video_id", ""),
                                 "task": task, "candidates": cands})
        if not final:
            failed += 1
            continue
        records.append({"item_index": pl["item"]["item_index"],
                        "video_id": pl["item"].get("video_id", ""),
                        "task": task, "prediction": final,
                        "source": "render" if cfg.direct else "dossier"})
    if failed:
        print(f"[text_dossier][WARN] {failed} render(s) empty/failed — left to the model.")
    if cfg.candidates_out:
        os.makedirs(os.path.dirname(cfg.candidates_out) or ".", exist_ok=True)
        with open(cfg.candidates_out, "w", encoding="utf-8") as f:
            for rec in cand_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_c = sum(len(r["candidates"]) for r in cand_records)
        print(f"[text_dossier] wrote {len(cand_records)} candidate pool(s) "
              f"({n_c} candidates) to {cfg.candidates_out}")
    return _write(records, out_path)


def _write(records: list[dict], out_path: str) -> list[dict]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    by_task: dict = {}
    for r in records:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print(f"[text_dossier] wrote {len(records)} override(s) to {out_path}")
    print(f"[text_dossier] by task: {by_task}")
    return records


def _load_preds(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["item_index"])] = rec
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _engine_args(a):
    """A namespace shaped like infer.parse_args() for infer.build_engine."""
    from types import SimpleNamespace
    return SimpleNamespace(
        adapter=a.adapter, model=a.model, backend=a.backend, attn_impl=a.attn_impl,
        max_lora_rank=a.max_lora_rank, tensor_parallel_size=a.tensor_parallel_size,
        gpu_memory_utilization=a.gpu_memory_utilization, device_map=a.device_map,
        max_model_len=a.max_model_len,
        num_frames=a.num_frames, temporal_num_frames=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="LoRA checkpoint dir.")
    src.add_argument("--model", help="Full/merged model path.")
    p.add_argument("--test-json", required=True,
                   help="items json (test.json or a val_gt.json).")
    p.add_argument("--pred", default="",
                   help="predictions jsonl (post-structural preferred) — supplies "
                        "the mcq consequence + bcq observations for the fact sheet.")
    p.add_argument("--videos-root", required=True)
    p.add_argument("--out", default="preds/text_dossier.jsonl")
    p.add_argument("--limit", type=int, default=0, help="first N videos (debug).")
    p.add_argument("--tasks", nargs="*", default=list(TARGET_TASKS),
                   help=f"target tasks to override (default: {list(TARGET_TASKS)}).")
    # engine knobs (mirror infer.py / temporal_grounding.py)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--device-map", default="")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--gen-batch", type=int, default=256,
                   help="requests per engine batch — bounds host RAM (each video-"
                        "attached request decord-decodes its clip at encode time).")
    # dossier knobs (DossierConfig)
    p.add_argument("--direct", action="store_true",
                   help="skip the analysis pass; render straight from the evidence "
                        "sheet — parity with the render-SFT-distilled model.")
    p.add_argument("--fact-policy", choices=["video-priority", "trust"],
                   default="video-priority", help="direct-mode fact-sheet header policy.")
    p.add_argument("--scene-probes", default="",
                   help="scene-attributes jsonl (claim_verify --scene-out) feeding "
                        "the evidence sheet's Scene line; empty = no Scene line.")
    # PROVE tier 3 — MBR candidate pool (consumed by track3.mbr_select).
    p.add_argument("--n-samples", type=int, default=1,
                   help="candidates per item: 1 greedy + N-1 sampled under fact "
                        "jitter (direct mode). 1 = off.")
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--candidates-out", default="",
                   help="write the {item_index, task, candidates} pool jsonl here.")
    p.add_argument("--seed", type=int, default=0, help="fact-jitter determinism.")
    p.add_argument("--no-render-video", action="store_true",
                   help="render text-only from the dossier (cheaper; further from "
                        "the SFT input distribution — gate on curated-val).")
    p.add_argument("--length-tol", type=float, default=0.3,
                   help="length budget = train-median*(1+tol); negative disables.")
    p.add_argument("--dossier-max-tokens", type=int, default=512)
    p.add_argument("--render-max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args()

    cfg = DossierConfig(
        tasks=tuple(a.tasks), render_with_video=not a.no_render_video,
        length_tol=a.length_tol, dossier_max_tokens=a.dossier_max_tokens,
        render_max_tokens=a.render_max_tokens, temperature=a.temperature,
        direct=a.direct, fact_policy=a.fact_policy,
        n_samples=a.n_samples, sample_temperature=a.sample_temperature,
        candidates_out=a.candidates_out, seed=a.seed)

    items_by_video = load_items_by_video(a.test_json)
    if a.limit:
        keep = list(items_by_video)[:a.limit]
        items_by_video = {k: items_by_video[k] for k in keep}
    preds = _load_preds(a.pred)
    scene_map = {}
    if a.scene_probes and os.path.exists(a.scene_probes):
        with open(a.scene_probes, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    scene_map[rec["video_id"]] = rec.get("scene") or []
    elif a.scene_probes:
        print(f"[text_dossier][WARN] scene probes {a.scene_probes} not found — "
              "no Scene line.")

    eargs = _engine_args(a)
    dossier_gen = DossierGenerator(eargs, a.dossier_max_tokens, a.temperature,
                                   batch_size=a.gen_batch)
    # The render/sample passes need different max_tokens/temperature; reuse the
    # same engine (a DossierGenerator view with its own RequestConfig).
    from swift import RequestConfig  # lazy

    def _gen_view(max_tokens: int, temperature: float):
        g = DossierGenerator.__new__(DossierGenerator)
        g.__dict__ = dict(dossier_gen.__dict__)
        g._cfg = RequestConfig(max_tokens=max_tokens, temperature=temperature)
        return g

    render_gen = _gen_view(a.render_max_tokens, a.temperature)
    sample_gen = _gen_view(a.render_max_tokens, a.sample_temperature) \
        if a.n_samples > 1 else None

    run(items_by_video, preds, a.videos_root, a.out, cfg, dossier_gen, render_gen,
        sample_gen=sample_gen, scene_map=scene_map)


if __name__ == "__main__":
    main()
