"""Referring-expression frame grounding for temporal_localization.

Writes a ``{item_index, start, end}`` jsonl for ``structural --temporal-override``.
Per temporal_localization item:

1. Parse the event phrase E from the question ("When does <E> occur?").
2. Prior window P = the same video's td (else cl) question window, typo-repaired.
3. Sample dense frames (about every ``stride`` s) with their timestamps.
4. Score each frame: P(first token == "Yes") to "Is <E> happening in this frame?".
5. Smooth; take the peak and the contiguous span above rel_thresh * peak.
6. Fuse with the prior:
     - peak inside P, or span overlaps P        -> keep P
     - peak outside P but not confident enough  -> keep P
     - otherwise                                -> relocate (prior width at the peak, or the span)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

from track3.structural import (
    load_items_by_video,
    normalize_ts,
    question_window,
    resolve_video,
)
from track3.tasks import extract_interval_obj, parse_timestamp

_WHEN = re.compile(r"when\s+does\s+(.+?)\s+occur\b", re.I)
_WHEN_ALT = re.compile(r"when\s+(?:is|are|do|did|was|were)\s+(.+?)[?.]", re.I)


def parse_event_phrase(question: str) -> str:
    """The referring expression E from a temporal_localization question.

    Falls back to the first question line, then a generic phrase; never empty.
    """
    first = (question or "").strip().splitlines()[0].strip() if question else ""
    m = _WHEN.search(first) or _WHEN_ALT.search(first)
    if m:
        return m.group(1).strip()
    return first.rstrip("?.").strip() or "the collision or anomaly"


def fmt_ts(sec: float) -> str:
    """Seconds -> "MM:SS.ff" (fractional seconds preserved)."""
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Interval IoU, matching track3/evaluate.py exactly (union 0 -> 0.0)."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(0.0, (e1 - s1) + (e2 - s2) - inter)
    return inter / union if union > 0 else 0.0


def smooth(xs: list[float], k: int = 3) -> list[float]:
    """Centered moving average over an odd window k (k<=1 -> identity)."""
    if k <= 1 or len(xs) <= 2:
        return list(xs)
    half = k // 2
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


@dataclass
class GroundConfig:
    # frame sampling
    stride: float = 0.4          # target seconds between sampled frames
    min_frames: int = 8
    max_frames: int = 48
    max_side: int = 448
    # detection
    smooth_k: int = 3
    rel_thresh: float = 0.5      # span = contiguous frames with score >= rel*peak
    min_width: float = 1.0       # floor on a detected span (s)
    # fusion with the prior
    conf_thresh: float = 0.5     # min peak P(Yes) to even consider relocating
    margin: float = 0.15         # peak must beat the best in-prior score by this
    agree_iou: float = 0.10      # det-span overlapping P this much -> P confirmed
    relocate_width: str = "prior"  # "prior" (center prior-width at peak) | "span"


def num_frames_for(duration: float, cfg: GroundConfig) -> int:
    """How many evenly-spaced frames to sample for a clip of ``duration`` s."""
    if duration <= 0:
        return cfg.min_frames
    n = round(duration / cfg.stride)
    return int(max(cfg.min_frames, min(cfg.max_frames, n)))


def ground_video(prior: tuple[float, float] | None,
                 timestamps: list[float],
                 scores: list[float],
                 duration: float,
                 cfg: GroundConfig) -> dict:
    """Final [start, end] (seconds) from per-frame scores + the prior. Pure, no IO.

    ``source``: prior_confirmed | prior_kept | relocated_priorwidth | relocated_span |
    detector_noprior | prior_only | empty.
    """
    base = {"peak_t": None, "peak_score": 0.0, "prior_score": 0.0,
            "det_start": None, "det_end": None}
    # no scores -> prior only
    if not scores or not timestamps or len(scores) != len(timestamps):
        if prior is not None:
            return {**base, "start": prior[0], "end": prior[1], "source": "prior_only"}
        return {**base, "start": 0.0, "end": 0.0, "source": "empty"}

    sm = smooth(scores, cfg.smooth_k)
    # peak = center of the top plateau
    mx = max(sm)
    plateau = [i for i, v in enumerate(sm) if v >= mx - 1e-9]
    peak_idx = plateau[len(plateau) // 2]
    peak_t, peak_score = timestamps[peak_idx], sm[peak_idx]

    # Contiguous span around the peak above a relative threshold.
    thr = peak_score * cfg.rel_thresh
    lo = hi = peak_idx
    while lo - 1 >= 0 and sm[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < len(sm) and sm[hi + 1] >= thr:
        hi += 1
    det_s, det_e = timestamps[lo], timestamps[hi]
    if det_e - det_s < cfg.min_width:               # widen a degenerate span
        half = cfg.min_width / 2.0
        det_s, det_e = peak_t - half, peak_t + half
    det_s = max(0.0, det_s)
    det_e = min(duration, det_e) if duration > 0 else det_e
    base.update(peak_t=peak_t, peak_score=peak_score, det_start=det_s, det_end=det_e)

    if prior is None:                                # rare (train edge); best effort
        return {**base, "start": det_s, "end": det_e, "source": "detector_noprior"}

    px, py = prior
    # best score inside the prior window
    in_prior = [sm[i] for i, t in enumerate(timestamps) if px <= t <= py]
    prior_score = max(in_prior) if in_prior else 0.0
    base["prior_score"] = prior_score

    # evidence points into the prior: keep it
    if (px <= peak_t <= py) or iou(prior, (det_s, det_e)) >= cfg.agree_iou:
        return {**base, "start": px, "end": py, "source": "prior_confirmed"}

    # peak outside the prior: relocate only if confident at the peak and unconvinced inside it
    confident = peak_score >= cfg.conf_thresh and (peak_score - prior_score) >= cfg.margin
    if not confident:
        return {**base, "start": px, "end": py, "source": "prior_kept"}

    if cfg.relocate_width == "prior" and (py - px) > 0:
        half = (py - px) / 2.0
        s, e = peak_t - half, peak_t + half
        s = max(0.0, s)
        e = min(duration, e) if duration > 0 else e
        return {**base, "start": s, "end": e, "source": "relocated_priorwidth"}
    return {**base, "start": det_s, "end": det_e, "source": "relocated_span"}


_SYS_FRAME = (
    "You are an expert traffic-surveillance video analyst. You are shown a SINGLE "
    "still frame extracted from a short traffic video. Decide whether the specific "
    "event described is happening, or its immediate impact is clearly visible, in "
    "THIS exact frame. Answer with only 'Yes' or 'No'."
)


def frame_prompt(event_phrase: str) -> str:
    return ("<image>\n"
            f"Event: {event_phrase}.\n"
            "Is this event happening, or is its immediate impact clearly visible, "
            "in this frame? Answer with only Yes or No.")


class FrameSampler:
    """Dense uniform frame extraction with timestamps (reuses track3.frames)."""

    def __init__(self, frames_root: str, cfg: GroundConfig):
        self.frames_root = frames_root
        self.cfg = cfg

    def __call__(self, video_path: str):
        from track3 import frames as frame_utils  # lazy: cv2
        duration = frame_utils.video_duration(video_path)
        n = num_frames_for(duration, self.cfg)
        out_dir = os.path.join(
            self.frames_root,
            f"{frame_utils.safe_name(os.path.basename(video_path))}_g{n}_s{self.cfg.max_side}")
        fr = frame_utils.extract_uniform_frames(video_path, out_dir, n, self.cfg.max_side)
        dur = fr.duration or duration or (fr.timestamps[-1] if fr.timestamps else 0.0)
        return fr.timestamps, fr.paths, dur


class FrameScorer:
    """Batched single-image P(Yes) scorer (one max_tokens=1 logprob pass)."""

    def __init__(self, args):
        from swift import InferRequest, RequestConfig  # lazy
        from track3.infer import build_engine, _supports_logprobs
        self._InferRequest = InferRequest
        self._RequestConfig = RequestConfig
        self.supported = _supports_logprobs(RequestConfig)
        self.engine, adapters = build_engine(args)
        self.extra = {}
        if adapters and args.backend == "vllm":
            from swift import AdapterRequest
            self.extra = {"adapter_request": AdapterRequest("track3", adapters[0])}

    def __call__(self, jobs: list[tuple[str, str]]) -> list[float]:
        from track3.infer import _first_token_dist
        if not self.supported:
            print("[temporal_grounding][WARN] backend has no logprobs/top_logprobs "
                  "(cannot score frames); emitting the pure td-window leak.")
            return []
        reqs = [self._InferRequest(
                    messages=[{"role": "system", "content": _SYS_FRAME},
                              {"role": "user", "content": frame_prompt(ev)}],
                    images=[path])
                for path, ev in jobs]
        cfg = self._RequestConfig(max_tokens=1, temperature=0,
                                  logprobs=True, top_logprobs=20)
        resp = self.engine.infer(reqs, cfg, **self.extra)
        out = []
        for r in resp:
            dist = _first_token_dist(r.choices[0], "yesno")
            out.append(float(dist.get("Yes", 0.0)))
        return out


def _prior_for(tasks: dict):
    """((X,Y) strings, (X,Y) seconds) of the td else cl window, or (None, None).

    Timestamps are typo-repaired (normalize_ts) and sorted by seconds.
    """
    for src in ("temporal_description", "causal_linkage"):
        for it in tasks.get(src, ()):
            w = question_window(it["question"])
            if w:
                a, b = normalize_ts(w[0]), normalize_ts(w[1])
                sa, sb = parse_timestamp(a), parse_timestamp(b)
                if sa > sb:
                    a, b, sa, sb = b, a, sb, sa
                return (a, b), (sa, sb)
    return None, None


def run(items_by_video: dict, videos_root: str, out_path: str, cfg: GroundConfig,
        sampler, scorer) -> list[dict]:
    """One override record per temporal_localization item.

    ``sampler(video_path) -> (timestamps, paths, duration)`` and
    ``scorer(jobs) -> [P(Yes)]`` are injected; all frames are scored in one batch.
    """
    plans, jobs = [], []
    for vid, tasks in items_by_video.items():
        tl_items = tasks.get("temporal_localization", ())
        if not tl_items:
            continue
        verb, prior = _prior_for(tasks)
        vpath = resolve_video(videos_root, vid)
        try:
            timestamps, paths, duration = sampler(vpath)
        except Exception as e:                      # missing/corrupt video -> leak
            print(f"[temporal_grounding][WARN] sample failed for {vid}: {e}")
            timestamps, paths, duration = [], [], 0.0
        for it in tl_items:
            ev = parse_event_phrase(it["question"])
            start = len(jobs)
            jobs.extend((p, ev) for p in paths)
            plans.append({"item": it, "verb": verb, "prior": prior,
                          "timestamps": timestamps, "duration": duration,
                          "span": (start, len(jobs))})

    scores = scorer(jobs) if jobs else []
    have_scores = len(scores) == len(jobs) and len(jobs) > 0

    records, by_source = [], defaultdict(int)
    for pl in plans:
        s = scores[pl["span"][0]:pl["span"][1]] if have_scores else []
        res = ground_video(pl["prior"], pl["timestamps"], s, pl["duration"], cfg)
        if res["source"] in ("prior_only", "prior_kept", "prior_confirmed") \
                and pl["verb"] is not None:
            start_str, end_str = pl["verb"]         # verbatim leak strings
        else:
            start_str, end_str = fmt_ts(res["start"]), fmt_ts(res["end"])
        by_source[res["source"]] += 1
        rec = {"item_index": pl["item"]["item_index"],
               "video_id": pl["item"]["video_id"],
               "start": start_str, "end": end_str, "source": res["source"],
               "peak_t": res["peak_t"], "peak_score": round(res["peak_score"], 4)}
        records.append(rec)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[temporal_grounding] wrote {len(records)} override(s) to {out_path}")
    print(f"[temporal_grounding] sources: {dict(by_source)}")
    return records


def _gt_interval(answer: str):
    obj = extract_interval_obj(answer)
    if not obj or "start" not in obj or "end" not in obj:
        return None
    try:
        return parse_timestamp(obj["start"]), parse_timestamp(obj["end"])
    except (KeyError, ValueError, TypeError):
        return None


def score_against_gt(records: list[dict], items_by_video: dict) -> dict:
    """mIoU of td-only / grounded / oracle vs GT, plus leak-correct (td IoU > 0.7)
    and leak-wrong (td IoU < 0.3) subsets."""
    gt, verb = {}, {}
    for vid, tasks in items_by_video.items():
        v, _ = _prior_for(tasks)
        for it in tasks.get("temporal_localization", ()):
            g = _gt_interval(it.get("answer", ""))
            if g is not None:
                gt[it["item_index"]] = g
                verb[it["item_index"]] = v
    if not gt:
        print("[temporal_grounding] no GT temporal answers; skipping scoring.")
        return {}

    td_all, gr_all, orc_all = [], [], []
    easy_td, easy_gr, hard_td, hard_gr = [], [], [], []
    for rec in records:
        idx = rec["item_index"]
        if idx not in gt:
            continue
        g = gt[idx]
        try:
            gr = (parse_timestamp(rec["start"]), parse_timestamp(rec["end"]))
        except (ValueError, TypeError):
            continue
        v = verb.get(idx)
        td = (parse_timestamp(v[0]), parse_timestamp(v[1])) if v else gr
        i_td, i_gr = iou(g, td), iou(g, gr)
        td_all.append(i_td)
        gr_all.append(i_gr)
        orc_all.append(max(i_td, i_gr))
        if i_td > 0.7:
            easy_td.append(i_td); easy_gr.append(i_gr)
        elif i_td < 0.3:
            hard_td.append(i_td); hard_gr.append(i_gr)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "n": len(td_all),
        "td_only_miou": mean(td_all),
        "grounded_miou": mean(gr_all),
        "oracle_miou": mean(orc_all),
        "leak_correct_n": len(easy_td),
        "leak_correct_grounded_miou": mean(easy_gr),  # must stay ≈ td (don't break easy)
        "leak_wrong_n": len(hard_td),
        "leak_wrong_td_miou": mean(hard_td),
        "leak_wrong_grounded_miou": mean(hard_gr),    # the relocation payoff
    }
    print("[temporal_grounding] ==== measurement vs GT ====")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    delta = metrics["grounded_miou"] - metrics["td_only_miou"]
    print(f"  {'grounded - td (Δ)':32s} {delta:+.4f}")
    return metrics


def load_train_items_by_video(train_dir: str, limit: int = 0) -> dict:
    by_video: dict = defaultdict(lambda: defaultdict(list))
    for task in ("temporal_localization", "temporal_description", "causal_linkage"):
        path = os.path.join(train_dir, f"{task}.json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path, encoding="utf-8"))
        for it in (doc.get("items", doc) if isinstance(doc, dict) else doc):
            rec = dict(it)
            rec.setdefault("task_type", task)
            rec.setdefault("item_index", rec.get("video_id"))
            by_video[rec["video_id"]][task].append(rec)
    if limit:
        keep = list(by_video)[:limit]
        by_video = {k: by_video[k] for k in keep}
    return by_video


def _engine_args(a):
    """A namespace shaped like infer.parse_args() for infer.build_engine."""
    from types import SimpleNamespace
    return SimpleNamespace(
        adapter=a.adapter, model=a.model, backend=a.backend, attn_impl=a.attn_impl,
        max_lora_rank=a.max_lora_rank, tensor_parallel_size=a.tensor_parallel_size,
        gpu_memory_utilization=a.gpu_memory_utilization, device_map=a.device_map,
        max_model_len=a.max_model_len,
        # one image per request
        num_frames=1, temporal_num_frames=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="LoRA checkpoint dir.")
    src.add_argument("--model", help="Full/merged model path.")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--test-json", help="items json (test.json or a val_gt.json).")
    gsrc.add_argument("--train-dir", help="data/train dir: audit on auto GT.")
    p.add_argument("--videos-root", required=True)
    p.add_argument("--out", default="preds/temporal_grounding.jsonl")
    p.add_argument("--frames-root", default="data/frames_grounding")
    p.add_argument("--limit", type=int, default=0, help="first N videos (debug/audit).")
    p.add_argument("--score", action="store_true",
                   help="also score vs GT temporal answers (val_gt / --train-dir).")
    # engine knobs (mirror infer.py)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--device-map", default="")
    p.add_argument("--max-model-len", type=int, default=4096)
    # grounding knobs (GroundConfig)
    p.add_argument("--stride", type=float, default=0.4)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--smooth-k", type=int, default=3)
    p.add_argument("--rel-thresh", type=float, default=0.5)
    p.add_argument("--min-width", type=float, default=1.0)
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--agree-iou", type=float, default=0.10)
    p.add_argument("--relocate-width", choices=["prior", "span"], default="prior")
    a = p.parse_args()

    cfg = GroundConfig(
        stride=a.stride, min_frames=a.min_frames, max_frames=a.max_frames,
        max_side=a.max_side, smooth_k=a.smooth_k, rel_thresh=a.rel_thresh,
        min_width=a.min_width, conf_thresh=a.conf_thresh, margin=a.margin,
        agree_iou=a.agree_iou, relocate_width=a.relocate_width)

    if a.train_dir:
        items_by_video = load_train_items_by_video(a.train_dir, a.limit)
    else:
        items_by_video = load_items_by_video(a.test_json)
        if a.limit:
            keep = list(items_by_video)[:a.limit]
            items_by_video = {k: items_by_video[k] for k in keep}

    sampler = FrameSampler(a.frames_root, cfg)
    scorer = FrameScorer(_engine_args(a))
    records = run(items_by_video, a.videos_root, a.out, cfg, sampler, scorer)

    if a.score or a.train_dir:
        score_against_gt(records, items_by_video)


if __name__ == "__main__":
    main()
