"""Whole-video Yes/No probing: claim verification + scene attributes.

  --candidates  split each candidate into atomic claims, probe P(first token == "Yes")
                per claim and write one mean verify score per candidate (mbr_select).
  --scene-out   probe the closed scene vocabulary per clip (text_dossier --scene-probes).

Run (claims + scene in one engine spin-up)::

    python -m track3.claim_verify --model <merged> --videos-root <root> \
        --candidates preds/candidates.jsonl --out preds/verify.jsonl \
        --test-json data/test/test.json --scene-out preds/scene_probes.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from track3.evidence import SCENE_PROBE_QUESTIONS, pick_scene_from_probes
from track3.structural import resolve_video

_SENT = re.compile(r"(?<=[.!?])\s+")
_LEAD_TOKEN = re.compile(r"^\s*(?:yes|no|[A-D])\b[.,:)]?\s*", re.IGNORECASE)
_YESNO_SUFFIX = " Answer with only Yes or No."


def decompose_claims(text: str, max_claims: int = 8) -> list[str]:
    """Deduped sentences with >= 4 words (lead Yes/No/letter stripped), at most ``max_claims``."""
    out: list[str] = []
    seen: set[str] = set()
    body = _LEAD_TOKEN.sub("", (text or "").strip(), count=1)
    for sent in _SENT.split(body):
        sent = sent.strip()
        if len(sent.split()) < 4:
            continue
        key = re.sub(r"\W+", " ", sent).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sent)
        if len(out) >= max_claims:
            break
    return out


def claim_probe_question(claim: str) -> str:
    """The probe stem for one claim, in the bcq register."""
    claim = (claim or "").strip().rstrip(".")
    return (f'Based on the video, is the following statement true? "{claim}."'
            + _YESNO_SUFFIX)


def scene_probe_jobs() -> list[tuple[tuple[str, str], str]]:
    """[((group, value), probe question)] over the closed scene vocabulary."""
    return [(key, q + _YESNO_SUFFIX) for key, q in SCENE_PROBE_QUESTIONS.items()]


def aggregate_verify(claim_probs: list[float]) -> float:
    """Mean claim P(Yes); neutral 0.5 when there are no claims."""
    return sum(claim_probs) / len(claim_probs) if claim_probs else 0.5


def verify_candidates(records: list[dict], prob_fn) -> list[dict]:
    """Per-record verify scores; ``prob_fn([{video_id, question}]) -> [P(Yes)]`` is injected.

    Probes are deduplicated per (video_id, claim).
    """
    probe_key_to_pos: dict[tuple[str, str], int] = {}
    jobs: list[dict] = []
    plans = []       # (rec, [[probe positions] per candidate])
    for rec in records:
        vid = rec.get("video_id", "")
        cand_positions = []
        for cand in rec.get("candidates", []):
            positions = []
            for claim in decompose_claims(cand):
                key = (vid, claim.lower())
                if key not in probe_key_to_pos:
                    probe_key_to_pos[key] = len(jobs)
                    jobs.append({"video_id": vid,
                                 "question": claim_probe_question(claim)})
                positions.append(probe_key_to_pos[key])
            cand_positions.append(positions)
        plans.append((rec, cand_positions))

    probs = prob_fn(jobs) if jobs else []
    out = []
    for rec, cand_positions in plans:
        scores = [aggregate_verify([probs[p] for p in positions])
                  for positions in cand_positions]
        out.append({"item_index": rec["item_index"],
                    "video_id": rec.get("video_id", ""), "scores": scores})
    if records:
        print(f"[verify] {len(jobs)} unique claim probe(s) over {len(records)} "
              f"item(s) ({sum(len(r.get('candidates', [])) for r in records)} candidates)")
    return out


def probe_scene(video_ids: list[str], prob_fn, min_p: float = 0.5) -> list[dict]:
    """Per-video scene attributes from the closed-vocabulary probes."""
    probes = scene_probe_jobs()
    jobs = [{"video_id": vid, "question": q}
            for vid in video_ids for _, q in probes]
    probs = prob_fn(jobs) if jobs else []
    out = []
    for i, vid in enumerate(video_ids):
        scores = {key: probs[i * len(probes) + k] for k, (key, _) in enumerate(probes)}
        out.append({"video_id": vid,
                    "scene": list(pick_scene_from_probes(scores, min_p=min_p))})
    return out


class VideoYesNoScorer:
    """Batched whole-video P(first token == "Yes") in the bcq register.

    Requests go in chunks of ``batch_size`` because each one decodes its clip into host
    RAM; a failed chunk yields neutral 0.5 for its jobs.
    """

    def __init__(self, args, videos_root: str, batch_size: int = 256):
        from swift import InferRequest, RequestConfig  # lazy
        from track3.infer import build_engine
        self._InferRequest = InferRequest
        self._cfg = RequestConfig(max_tokens=1, temperature=0,
                                  logprobs=True, top_logprobs=20)
        self.engine, adapters = build_engine(args)
        self.extra = {}
        if adapters and args.backend == "vllm":
            from swift import AdapterRequest
            self.extra = {"adapter_request": AdapterRequest("track3", adapters[0])}
        self.videos_root = videos_root
        self.batch_size = max(1, batch_size)

    def __call__(self, jobs: list[dict]) -> list[float]:
        from track3.infer import _first_token_dist
        from track3.tasks import build_user_prompt, get_task
        if not jobs:
            return []
        spec = get_task("bcq")
        out: list[float] = []
        for lo in range(0, len(jobs), self.batch_size):
            sub = jobs[lo:lo + self.batch_size]
            reqs = [self._InferRequest(
                messages=[{"role": "system", "content": spec.system},
                          {"role": "user",
                           "content": build_user_prompt(spec, j["question"])}],
                videos=[resolve_video(self.videos_root, j["video_id"])]) for j in sub]
            try:
                resp = self.engine.infer(reqs, self._cfg, **self.extra)
            except Exception as e:
                print(f"[verify][WARN] probe chunk failed ({e}); neutral 0.5 for "
                      f"{len(sub)} probe(s).")
                out.extend([0.5] * len(sub))
                continue
            for r in resp:
                dist = _first_token_dist(r.choices[0], "yesno")
                out.append(float(dist.get("Yes", 0.5 if not dist else 0.0)))
            if len(jobs) > self.batch_size:
                print(f"[verify] probed {min(lo + self.batch_size, len(jobs))}"
                      f"/{len(jobs)}", flush=True)
        return out


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="LoRA checkpoint dir.")
    src.add_argument("--model", help="Full/merged model path.")
    p.add_argument("--videos-root", required=True)
    # claims mode
    p.add_argument("--candidates", default="",
                   help="candidates jsonl (text_dossier --candidates-out).")
    p.add_argument("--out", default="preds/verify.jsonl",
                   help="verify-scores jsonl (claims mode).")
    # scene mode
    p.add_argument("--test-json", default="",
                   help="items json: the clips to scene-probe.")
    p.add_argument("--scene-out", default="",
                   help="scene-attributes jsonl for text_dossier --scene-probes.")
    p.add_argument("--scene-min-p", type=float, default=0.5)
    # engine knobs (mirror text_dossier.py)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--device-map", default="")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--probe-batch", type=int, default=256,
                   help="probes per engine batch; bounds host RAM (each video-"
                        "attached request decord-decodes its clip at encode time).")
    a = p.parse_args()
    if not a.candidates and not a.scene_out:
        p.error("nothing to do: pass --candidates and/or --scene-out")
    if a.scene_out and not a.test_json:
        p.error("--scene-out needs --test-json for the clip list")

    test_items = []
    if a.test_json:
        data = json.load(open(a.test_json, encoding="utf-8"))
        test_items = data.get("items", data) if isinstance(data, dict) else data

    from track3.text_dossier import _engine_args
    scorer = VideoYesNoScorer(_engine_args(a), a.videos_root, a.probe_batch)

    if a.candidates:
        records = _load_jsonl(a.candidates)
        out = verify_candidates(records, scorer)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[verify] wrote {len(out)} verify record(s) to {a.out}")

    if a.scene_out:
        vids = sorted({it["video_id"] for it in test_items})
        missing = [v for v in vids
                   if not os.path.exists(resolve_video(a.videos_root, v))]
        if missing:
            print(f"[verify][WARN] {len(missing)} video(s) missing, skipped.")
        vids = [v for v in vids if v not in set(missing)]
        out = probe_scene(vids, scorer, min_p=a.scene_min_p)
        os.makedirs(os.path.dirname(a.scene_out) or ".", exist_ok=True)
        with open(a.scene_out, "w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = sum(1 for r in out if r["scene"])
        print(f"[verify] scene attributes for {n}/{len(out)} clip(s) -> {a.scene_out}")


if __name__ == "__main__":
    main()
