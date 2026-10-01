"""Option-anchored mcq_openended answers, written as a ``structural --text-override`` jsonl.

The GT ``"X. <reason>"`` largely restates the chosen option text, so:

  * --mode anchor: emit ``"X. <chosen option text>"`` (deterministic, no model).
  * --mode render: one sentence anchored on the chosen option (text_dossier generator).

The letter comes from our own prediction; items without a parseable letter/option
are skipped and keep the model's text.
"""
from __future__ import annotations

import argparse
import json
import os

from track3.structural import load_items_by_video, question_options, resolve_video
from track3.tasks import extract_letter, get_task

# Train-GT median mcq_openended answer length (render budget).
LENGTH_TARGET = 98


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
    """Deterministic answer: ``"X. <option as a sentence>"``."""
    return f"{letter}. {normalize_clause(option_text)}".strip()


def chosen_letter_option(item: dict, preds: dict) -> tuple[str, str] | tuple[None, None]:
    """(letter, option_text) from our prediction + the question; (None, None) if unresolved."""
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
    """One deterministic option-anchored override per mcq_openended item."""
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
    """Render user turn anchored on the chosen option."""
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
    """Option-anchored one-sentence render per mcq_openended item.

    ``gen([{system, user, video}]) -> [str]`` is injected; unresolved items keep the
    model's text.
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
                   help="predictions jsonl (post-structural preferred); the chosen "
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
