"""Option-anchored mcq_openended — close the one task a competitor beats us on.

ISOLATED, REMOVABLE add-on (like ``text_dossier`` / ``text_mbr`` / ``temporal_
grounding``): one module + ``scripts/mcqoe_anchor.sh``. It touches NO existing file
— it only *produces* a ``{item_index, task, prediction}`` override jsonl (the
``text_dossier`` shape) that the already-present ``track3.structural
--text-override`` hook consumes. Removal =
``git rm track3/mcqoe_anchor.py track3/test_mcqoe_anchor.py scripts/mcqoe_anchor.sh``.

----------------------------------------------------------------------------
WHY (the leaderboard + data this followed)
----------------------------------------------------------------------------
On the public board we lead every column EXCEPT mcq_openended (ours ~0.827 vs the
top competitor ~0.915 — the single per-task deficit). Measured on 3,670 train
mcq_openended items, the GT answer ``"X. <reason>"`` is structurally:

  * HALF the chosen OPTION's text: the reason covers the chosen option at median
    0.50 vs only 0.25 for the best distractor (2x discrimination); 31% are a
    near-copy (>=70%) of the option. e.g. GT == option ==
    "Vehicles traveling at excessive speeds and drivers losing control."
  * PLUS grounded elaboration: ~62% of the reason's words are NOT in the option
    (one extra clause grounded in the video); reason median 95 chars, option 72.

So half the answer is *given* in the question's option text — the same "answer-in-
the-question" structure the pipeline already exploits for temporal / bcq / cross-
task, applied to the ONE task that still free-generates (rewrite-only) and drifts.

Two variants (A/B on the board; both FAQ-clean — option text is in the released
question, the letter is our own high-accuracy mcq prediction):

  * MODE=anchor (V1, deterministic, NO model): emit ``"X. <chosen option text>"``.
    Precision-maxed floor — captures the option backbone the reference restates.
  * MODE=render (V2, model): a one-sentence render *anchored* on the chosen option
    (option content + one grounded clause), capped to the train median ~95 chars,
    mirroring the GT structure. Reuses the text_dossier generator; the option is
    injected into the RENDER prompt (an already-OOD fact-injection path), NOT the
    SFT prompt — so it never re-triggers the F6 prompt-edit register drift.

Letter↔body consistency: the leading letter is sourced from PRED (post-structural
preferred), where rule 4.3 has already aligned mcq<->mcq_openended, so the final
4.3 pass is a no-op on the anchored answer. Items without a parseable letter/option
are skipped (the override is partial-safe -> they keep the model's own text). Gate
on the BOARD.

Pure logic (option parse / normalise / assembly / prompt) has NO heavy deps and is
unit-tested on a CPU dev box; only MODE=render imports the generator lazily.
"""
from __future__ import annotations

import argparse
import json
import os

from track3.structural import load_items_by_video, question_options, resolve_video
from track3.tasks import extract_letter, get_task

# Train-GT median answer length for mcq_openended (data.md §2) — the render budget.
LENGTH_TARGET = 98


# ---------------------------------------------------------------------------
# Pure logic — no ms-swift / torch (unit-testable on the dev box)
# ---------------------------------------------------------------------------


def normalize_clause(text: str) -> str:
    """Make an option fragment read as a standalone justification sentence."""
    text = (text or "").strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def build_anchor(letter: str, option_text: str) -> str:
    """The deterministic V1 answer: ``"X. <option as a sentence>"``."""
    return f"{letter}. {normalize_clause(option_text)}".strip()


def chosen_letter_option(item: dict, preds: dict) -> tuple[str, str] | tuple[None, None]:
    """(letter, option_text) for an mcq_openended item from PRED + the question.

    The letter is read from our prediction (post-structural preferred, already 4.3-
    aligned); the option text is the *given* option for that letter. Returns
    ``(None, None)`` when the prediction has no parseable letter or the letter is
    not among the question's options (the caller then leaves the item to the model).
    """
    rec = preds.get(str(item.get("item_index", "")))
    if not rec:
        return None, None
    letter = extract_letter(rec.get("prediction") or "")
    if not letter:
        return None, None
    opts = question_options(item.get("question", ""), normalize=False)
    if letter not in opts:
        return None, None
    return letter, opts[letter]


def anchored_records(items_by_video: dict, preds: dict) -> list[dict]:
    """V1: one deterministic option-anchored override per mcq_openended item."""
    out = []
    spec = get_task("mcq_openended")
    for vid, tasks in items_by_video.items():
        for it in tasks.get("mcq_openended", ()):
            letter, opt = chosen_letter_option(it, preds)
            if not letter:
                continue
            text = spec.submission(build_anchor(letter, opt))
            out.append({"item_index": it["item_index"],
                        "video_id": it.get("video_id", ""),
                        "task": "mcq_openended", "prediction": text,
                        "source": "mcqoe_anchor"})
    return out


SYS_RENDER = (
    "You are an expert traffic-accident analyst. Justify the verified correct option "
    "in ONE concise, factual sentence grounded in the video — restate its content and "
    "add one grounded detail. Do not hedge or list other options."
)


def render_prompt(question: str, letter: str, option_text: str,
                  with_video: bool = True) -> str:
    """V2 user turn: anchor the render on the chosen option (NOT the SFT prompt)."""
    lines = ["<video>"] if with_video else []
    lines += [question.strip(),
              f"The verified correct option is {letter}) {option_text}",
              "Justify this option in one concise grounded sentence."]
    return "\n".join(lines)


def calibrate(text: str, budget: int) -> str:
    """Trim to <= budget chars on a sentence boundary (reuses the dossier helper)."""
    from track3.text_dossier import calibrate_length
    return calibrate_length(text, budget)


def render_records(items_by_video: dict, preds: dict, videos_root: str,
                   gen, with_video: bool = True, length_tol: float = 0.0,
                   exists_fn=os.path.exists) -> list[dict]:
    """V2: option-anchored one-sentence render per mcq_openended item.

    ``gen(jobs)`` takes ``[{system,user,video}]`` and returns one string per job
    (injected — the real text_dossier.DossierGenerator, or a stub in tests). Items
    without a parseable letter/option, or a missing video, fall back to the model.
    """
    spec = get_task("mcq_openended")
    budget = round(LENGTH_TARGET * (1.0 + length_tol)) if length_tol >= 0 else 0
    plans, jobs = [], []
    for vid, tasks in items_by_video.items():
        vpath = resolve_video(videos_root, vid)
        has_video = (not with_video) or exists_fn(vpath)
        for it in tasks.get("mcq_openended", ()):
            letter, opt = chosen_letter_option(it, preds)
            if not letter:
                continue
            jobs.append({"system": SYS_RENDER,
                         "user": render_prompt(it.get("question", ""), letter, opt,
                                               with_video and has_video),
                         "video": vpath if (with_video and has_video) else None})
            plans.append({"item": it, "letter": letter})

    texts = gen(jobs) if jobs else []
    if len(texts) != len(jobs):
        print(f"[mcqoe_anchor][WARN] generator returned {len(texts)} for {len(jobs)} "
              "jobs; emitting no render override (items keep the model's text).")
        return []
    out = []
    for pl, text in zip(plans, texts):
        body = (text or "").strip()
        if not body:
            continue  # leave to the model
        # keep the explanation, force the verified leading letter, cap length.
        from track3.text_dossier import _prefix_token
        ans = calibrate(_prefix_token(body, pl["letter"]), budget)
        ans = spec.submission(ans)
        out.append({"item_index": pl["item"]["item_index"],
                    "video_id": pl["item"].get("video_id", ""),
                    "task": "mcq_openended", "prediction": ans,
                    "source": "mcqoe_render"})
    return out


def write_override(records: list[dict], out_path: str) -> list[dict]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[mcqoe_anchor] wrote {len(records)} mcq_openended override(s) to {out_path}")
    return records


def load_preds(path: str) -> dict:
    out = {}
    if path and os.path.exists(path):
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
    from types import SimpleNamespace
    return SimpleNamespace(
        adapter=a.adapter, model=a.model, backend=a.backend, attn_impl=a.attn_impl,
        max_lora_rank=a.max_lora_rank, tensor_parallel_size=a.tensor_parallel_size,
        gpu_memory_utilization=a.gpu_memory_utilization, device_map=a.device_map,
        max_model_len=a.max_model_len, num_frames=a.num_frames, temporal_num_frames=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-json", required=True)
    p.add_argument("--pred", required=True,
                   help="predictions jsonl (post-structural preferred) — the chosen "
                        "mcq letter source.")
    p.add_argument("--out", default="preds/mcqoe_anchor.jsonl")
    p.add_argument("--mode", choices=["anchor", "render"], default="anchor",
                   help="anchor=deterministic 'X. <option>' (no model); "
                        "render=option-anchored one-sentence render (needs a model).")
    p.add_argument("--limit", type=int, default=0)
    # render-only model knobs
    src = p.add_mutually_exclusive_group()
    src.add_argument("--adapter")
    src.add_argument("--model")
    p.add_argument("--videos-root", default="")
    p.add_argument("--no-render-video", action="store_true")
    p.add_argument("--length-tol", type=float, default=0.0)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--device-map", default="")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--num-frames", type=int, default=16)
    a = p.parse_args()

    items_by_video = load_items_by_video(a.test_json)
    if a.limit:
        keep = list(items_by_video)[:a.limit]
        items_by_video = {k: items_by_video[k] for k in keep}
    preds = load_preds(a.pred)

    if a.mode == "anchor":
        records = anchored_records(items_by_video, preds)
    else:
        if not (a.model or a.adapter) or not a.videos_root:
            raise SystemExit("--mode render needs --model/--adapter and --videos-root.")
        from track3.text_dossier import DossierGenerator
        gen = DossierGenerator(_engine_args(a), max_new_tokens=128, temperature=0.0)
        records = render_records(items_by_video, preds, a.videos_root, gen,
                                 with_video=not a.no_render_video,
                                 length_tol=a.length_tol)
    write_override(records, a.out)


if __name__ == "__main__":
    main()
