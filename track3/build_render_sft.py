"""Evidence-Sheet -> Answer SFT distillation dataset builder (PROVE Phase 1;
strategy C: GT facts + noise injection). Distils the reference-register answer
distribution into the model through the exact evidence-grounded interface used at
inference.

ISOLATED, REMOVABLE add-on (like text_dossier / mcqoe_anchor): one module +
track3/test_build_render_sft.py + scripts/prove/phase1_prepare.sh. It only READS the
train GT (data/train/<task>.json) and WRITES an ms-swift SFT jsonl (same shape as
data/processed/train.jsonl). Nothing in the current pipeline changes.

----------------------------------------------------------------------------
STRATEGY (C) — GT facts + DAgger-style noise injection (PROVE.md §1 tier 1)
----------------------------------------------------------------------------
Each SFT example is  (video + evidence sheet) -> GT narrative answer.  The sheet is
the CANONICAL Evidence Sheet v2 (track3/evidence.py — one schema shared verbatim
with the test-time renderer), built from the TRAIN GT; every prediction-derived
field is corrupted with a fixed probability so the training fact-noise MATCHES the
test fact-noise:

  * event_phrase, window, cast : parsed from the QUESTIONS -> exact at test -> NEVER
                                 corrupted.
  * cause / consequence        : our mcq-family PREDICTIONS at test (~0.85-0.93 acc)
    (chosen option texts)        -> each replaced by a DISTRACTOR option with prob
                                 ``p_consequence``.
  * scene attributes           : W1 probes at test -> one attribute swapped within
                                 its group with prob ``p_scene``.
  * observations (bcq_oe)      : our bcq_oe PREDICTIONS at test -> each dropped with
                                 prob ``p_obs_drop`` (optionally Yes/No-flipped).

Crucially the TARGET is ALWAYS the true GT answer, even when a fact was corrupted:
the model is taught to DOWN-WEIGHT an unreliable fact and fall back on the video
(the prompt states the video is the ground truth). This is a static, dataset-level
approximation of scheduled sampling / DAgger.

At train time mix this with a replay of the original data/processed/train.jsonl
(mix_replay keeps ALL base records) into ONE LoRA adapter so closed-task ability is
preserved (unified system).

Pure logic (evidence assembly / noise / prompt / record) has NO ms-swift / torch
deps and is unit-tested on a CPU dev box.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from track3.evidence import (
    SYS_RENDER,
    EvidenceSheet,
    classify_stem,
    corrupt_scene,
    extract_cast,
    mine_scene_attributes,
    render_evidence_prompt,
)
from track3.structural import normalize_ts, question_options, question_window, resolve_video
from track3.temporal_grounding import parse_event_phrase

# The 6 BERTScore narrative tasks whose reference register we distil.
NARRATIVE_TASKS = (
    "temporal_description",
    "causal_linkage",
    "open_qa",
    "video_summarization",
    "scene_description",
    "bcq_openended",
)
# Tasks read ONLY to build the evidence sheet (never emitted as targets on their own).
_FACT_SOURCE_TASKS = ("temporal_localization", "mcq", "mcq_openended", "bcq",
                      "temporal_description", "causal_linkage", "scene_description")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    p_consequence: float = 0.3   # prob: replace a chosen mcq option with a distractor
    p_scene: float = 0.25        # prob: swap one scene attribute within its group
    p_obs_drop: float = 0.3      # per-observation drop prob
    p_obs_flip: float = 0.0      # prob: flip an observation's leading Yes<->No
    max_obs: int = 2             # observations kept before dropping


@dataclass
class BuildConfig:
    tasks: tuple[str, ...] = NARRATIVE_TASKS
    with_video: bool = True
    fact_policy: str = "video-priority"   # or "trust"
    per_task_cap: int = 0                 # 0 = keep all
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 0


# ---------------------------------------------------------------------------
# Evidence sheet (GT-derived, with noise) — pure, unit-testable
# ---------------------------------------------------------------------------

_LEAD_YESNO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def flip_yesno(text: str) -> str:
    """Flip a leading Yes<->No (keep the explanation) — for p_obs_flip corruption."""
    m = _LEAD_YESNO.match(text or "")
    if not m:
        return text
    repl = "No" if m.group(1).lower() == "yes" else "Yes"
    return repl + text[m.end():]


def _event_phrase(video_tasks: dict) -> str:
    for it in video_tasks.get("temporal_localization", ()):
        ev = parse_event_phrase(it.get("question", ""))
        if ev:
            return ev
    return "the collision or anomaly"


def _safe_window(question: str):
    """question_window that never raises on malformed train timestamps (e.g. '01.30.00')."""
    try:
        return question_window(question or "")
    except Exception:
        return None


def _window(video_tasks: dict, item: dict) -> tuple[str, str] | None:
    """The item's own [X,Y] window (td/cl carry it in the question), else the video's."""
    w = _safe_window(item.get("question", ""))
    if not w:
        for src in ("temporal_description", "causal_linkage"):
            for it in video_tasks.get(src, ()):
                w = _safe_window(it.get("question", ""))
                if w:
                    break
            if w:
                break
    return (normalize_ts(w[0]), normalize_ts(w[1])) if w else None


def _cast(video_tasks: dict) -> tuple[str, ...]:
    """Vehicle/agent descriptors over ALL of the video's question texts (sorted task
    order so the sheet is deterministic regardless of dict insertion order)."""
    qs = [it.get("question", "") for task in sorted(video_tasks)
          for it in video_tasks[task]]
    return extract_cast(qs)


def _scene(video_tasks: dict, rng: random.Random,
           noise: NoiseConfig) -> tuple[tuple[str, ...], bool]:
    """Scene attributes mined from the SD ground truth, noise-corrupted w/ p_scene."""
    for it in video_tasks.get("scene_description", ()):
        mined = mine_scene_attributes(it.get("answer") or "")
        if mined:
            return corrupt_scene(mined, rng, noise.p_scene)
    return (), False


def _mcq_facts(video_tasks: dict, rng: random.Random,
               noise: NoiseConfig) -> tuple[str, str, bool]:
    """(cause, consequence, corrupted) from the GT-chosen options of BOTH mcq-family
    items. Root-cause/fault stems feed 'cause', the rest 'consequence'; duplicate
    option texts (the shuffled twins, data.md §5.3) dedupe to one line. With prob
    ``p_consequence`` a chosen text is replaced by a distractor from its question."""
    cause = consequence = ""
    corrupted = False
    seen: set[str] = set()
    for src in ("mcq", "mcq_openended"):
        for it in video_tasks.get(src, ()):
            opts = question_options(it.get("question", ""), normalize=False)
            letter = (it.get("answer") or "").strip()[:1].upper()
            if letter not in opts:
                continue
            key = re.sub(r"\W+", " ", opts[letter]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            text = opts[letter]
            distractors = [v for k, v in opts.items() if k != letter]
            if distractors and rng.random() < noise.p_consequence:
                text = rng.choice(distractors)
                corrupted = True
            if classify_stem(it.get("question", "")) == "cause" and not cause:
                cause = text
            elif not consequence:
                consequence = text
    return cause, consequence, corrupted


def _observations(video_tasks: dict, rng: random.Random, noise: NoiseConfig,
                  skip_item: dict | None = None) -> list[str]:
    """Up to max_obs bcq_oe GT answers, each dropped with p_obs_drop / flipped w/ p_obs_flip."""
    out = []
    for it in video_tasks.get("bcq_openended", ()):
        if skip_item is not None and it is skip_item:
            continue  # don't feed a bcq_oe item its own answer as an 'observation'
        ans = (it.get("answer") or "").strip()
        if not ans:
            continue
        if len(out) >= noise.max_obs:
            break
        if rng.random() < noise.p_obs_drop:
            continue
        if noise.p_obs_flip and rng.random() < noise.p_obs_flip:
            ans = flip_yesno(ans)
        out.append(ans)
    return out


def build_evidence(video_tasks: dict, item: dict, rng: random.Random,
                   noise: NoiseConfig, task: str = "") -> EvidenceSheet:
    cause, consequence, mcq_corrupted = _mcq_facts(video_tasks, rng, noise)
    scene, scene_corrupted = _scene(video_tasks, rng, noise)
    skip = item if task == "bcq_openended" else None
    return EvidenceSheet(
        video_id=item.get("video_id", ""),
        event_phrase=_event_phrase(video_tasks),
        window=_window(video_tasks, item),
        cast=_cast(video_tasks),
        scene=scene,
        cause=cause,
        consequence=consequence,
        observations=_observations(video_tasks, rng, noise, skip_item=skip),
        corrupted=mcq_corrupted or scene_corrupted,
    )


def to_record(item: dict, task: str, facts: EvidenceSheet, vpath: str,
              cfg: BuildConfig) -> dict:
    """One ms-swift SFT record; TARGET is always the true GT answer."""
    user = render_evidence_prompt(task, item.get("question", ""), facts,
                                 cfg.with_video, cfg.fact_policy)
    return {
        "messages": [
            {"role": "system", "content": SYS_RENDER},
            {"role": "user", "content": user},
            {"role": "assistant", "content": item["answer"]},
        ],
        "videos": [vpath] if cfg.with_video else [],
        "task": task, "task_type": task,
        "video_id": item.get("video_id", ""),
        "answer": item["answer"],
        "fact_corrupted": facts.corrupted,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_train_by_video(train_dir: str, tasks) -> dict:
    """{video_id: {task: [items]}} over the target tasks + the fact-source tasks."""
    needed = set(tasks) | set(_FACT_SOURCE_TASKS)
    by_video: dict = defaultdict(lambda: defaultdict(list))
    for task in sorted(needed):
        path = os.path.join(train_dir, f"{task}.json")
        if not os.path.exists(path):
            print(f"[render_sft][WARN] no train file for '{task}' at {path} — skipped.")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", data) if isinstance(data, dict) else data
        for it in items:
            by_video[it["video_id"]][task].append(it)
    return by_video


def build(by_video: dict, videos_root: str, cfg: BuildConfig,
          exists_fn=os.path.exists) -> list[dict]:
    """Emit SFT records (target tasks only), noise-injected + per-task subsampled."""
    per_task: dict = defaultdict(list)
    skipped_novideo = failed = 0
    for vid in sorted(by_video):
        vt = by_video[vid]
        vpath = resolve_video(videos_root, vid)
        if cfg.with_video and not exists_fn(vpath):
            skipped_novideo += 1
            continue
        for task in cfg.tasks:
            for i, it in enumerate(vt.get(task, ())):
                if not (it.get("answer") or "").strip():
                    continue
                try:
                    # Per-item deterministic RNG: reproducible + stable per item.
                    irng = random.Random(f"{cfg.seed}:{vid}:{task}:{i}")
                    facts = build_evidence(vt, it, irng, cfg.noise, task=task)
                    per_task[task].append(to_record(it, task, facts, vpath, cfg))
                except Exception as e:  # a malformed train row must not kill the build
                    failed += 1
                    if failed <= 5:
                        print(f"[render_sft][WARN] skip {vid}/{task}#{i}: {type(e).__name__}: {e}")
    if skipped_novideo:
        print(f"[render_sft][WARN] {skipped_novideo} video(s) missing under {videos_root}"
              " — skipped.")
    if failed:
        print(f"[render_sft][WARN] {failed} item(s) skipped (malformed train rows).")

    rng = random.Random(cfg.seed)
    out: list[dict] = []
    for task in cfg.tasks:
        recs = per_task.get(task, [])
        rng.shuffle(recs)
        out.extend(recs[:cfg.per_task_cap] if cfg.per_task_cap else recs)
    return out


def mix_replay(render_records: list[dict], replay_path: str, ratio: float,
               out_path: str, seed: int) -> int:
    """Combine the FULL base SFT set with the render set into ONE training file. To
    prevent catastrophic forgetting, ALL base records are kept (replay must NEVER
    shrink the base) and the render set is downsampled to at most base/ratio, i.e.
    base:render >= ratio:1. Deterministic (seeded); drops the 'fact_corrupted' key.
    Returns the total record count written."""
    base = []
    with open(replay_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base.append(json.loads(line))
    rng = random.Random(seed + 1)
    clean = [{k: v for k, v in r.items() if k != "fact_corrupted"} for r in render_records]
    rng.shuffle(clean)
    n_render = min(len(clean), int(round(len(base) / ratio))) if ratio and ratio > 0 else len(clean)
    combined = base + clean[:n_render]          # ALL base preserved + render minority
    rng.shuffle(combined)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    frac = n_render / max(len(combined), 1)
    print(f"[render_sft] mixed ALL {len(base)} base + {n_render}/{len(clean)} render "
          f"(base:render>={ratio}:1, render {frac:.0%}) -> {out_path}  ({len(combined)} records)")
    return len(combined)


def write_jsonl(records: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_task: dict = defaultdict(int)
    corrupted: dict = defaultdict(int)
    for r in records:
        by_task[r["task"]] += 1
        corrupted[r["task"]] += int(r.get("fact_corrupted", False))
    print(f"[render_sft] wrote {len(records)} SFT record(s) to {out_path}")
    for t in sorted(by_task):
        n = by_task[t]
        print(f"[render_sft]   {t:22s} {n:5d}  (a fact corrupted: "
              f"{corrupted[t]}/{n} = {corrupted[t]/max(n,1):.0%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--videos-root", default=os.environ.get("TRAIN_VIDEOS_ROOT", ""),
                   help="train videos root (or set TRAIN_VIDEOS_ROOT).")
    p.add_argument("--out", default="data/processed/render_sft.jsonl")
    p.add_argument("--tasks", nargs="*", default=list(NARRATIVE_TASKS))
    p.add_argument("--per-task-cap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fact-policy", choices=["video-priority", "trust"],
                   default="video-priority")
    p.add_argument("--p-consequence", type=float, default=0.3)
    p.add_argument("--p-scene", type=float, default=0.25)
    p.add_argument("--p-obs-drop", type=float, default=0.3)
    p.add_argument("--p-obs-flip", type=float, default=0.0)
    p.add_argument("--max-obs", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="first N videos (debug).")
    p.add_argument("--no-require-video", action="store_true",
                   help="emit records even if the video file is absent (dev/dry-run).")
    # Replay-mix into a single training file (render:base = 1:replay-ratio).
    p.add_argument("--replay-jsonl", default="",
                   help="base SFT jsonl (data/processed/train.jsonl) to interleave.")
    p.add_argument("--replay-ratio", type=float, default=2.0)
    p.add_argument("--mixed-out", default="",
                   help="write the render+replay training file here (needs --replay-jsonl).")
    a = p.parse_args()

    cfg = BuildConfig(
        tasks=tuple(a.tasks), with_video=not a.no_video, fact_policy=a.fact_policy,
        per_task_cap=a.per_task_cap, seed=a.seed,
        noise=NoiseConfig(p_consequence=a.p_consequence, p_scene=a.p_scene,
                          p_obs_drop=a.p_obs_drop, p_obs_flip=a.p_obs_flip,
                          max_obs=a.max_obs))

    by_video = load_train_by_video(a.train_dir, cfg.tasks)
    if a.limit:
        keep = sorted(by_video)[:a.limit]
        by_video = {k: by_video[k] for k in keep}
    exists = (lambda _p: True) if a.no_require_video else os.path.exists
    records = build(by_video, a.videos_root, cfg, exists_fn=exists)
    write_jsonl(records, a.out)
    if a.mixed_out and a.replay_jsonl:
        mix_replay(records, a.replay_jsonl, a.replay_ratio, a.mixed_out, cfg.seed)


if __name__ == "__main__":
    main()
