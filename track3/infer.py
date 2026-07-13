"""Batched inference over the TAR ``test.json`` with a fine-tuned Qwen3-VL.

Builds a task-aware :class:`InferRequest` per test item (same system/user template
as training), runs the ms-swift engine (vLLM by default, transformers fallback),
applies **self-consistency voting** for the accuracy tasks, and writes a raw
predictions jsonl consumed by :mod:`track3.make_submission`.

Engine/adapter handling follows ms-swift's documented Python API
(``examples/infer/demo_lora.py``). Tested API surface: ms-swift ≥ 3.x.

Run: ``python -m track3.infer --adapter output/ckpt --test-json data/test/test.json``
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from track3 import frames as frame_utils
from track3.mcq_debias import merge_permuted_votes, permuted_question
from track3.structural import question_options, question_stem
from track3.tasks import (
    METRIC_ACC_LETTER,
    METRIC_ACC_YESNO,
    build_user_prompt,
    check_parseable,
    frame_plan,
    get_task,
    infer_task_key,
    strip_reasoning,
    vote_token,
)

# Closed-family tasks whose first generated token IS the decision (per the SFT
# targets: bcq -> "Yes"/"No", mcq -> bare letter, *_openended -> "Yes. ..."/"B. ...").
# Used by the W1 first-token logprob scoring path (method.md §9 W1).
CLOSED_TOKEN_KIND = {
    "bcq": "yesno",
    "bcq_openended": "yesno",
    "mcq": "letter",
    "mcq_openended": "letter",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="LoRA checkpoint dir (preferred).")
    src.add_argument("--model", help="Full model path (if not using LoRA).")
    p.add_argument("--test-json", required=True)
    p.add_argument("--videos-root", required=True)
    p.add_argument("--out", default="preds/test_pred.jsonl")
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--attn-impl", default="sdpa",
                   help="Attention for the transformers backend (sdpa needs no extra "
                        "package; flash_attn requires flash-attn installed).")
    p.add_argument("--max-lora-rank", type=int, default=64,
                   help="vLLM LoRA rank cap; must be >= the trained lora_rank.")
    # --- multi-GPU engine sharding --------------------------------------------
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="vLLM tensor parallelism: shard ONE model across N GPUs for "
                        "multi-GPU inference. Set to the CUDA_VISIBLE_DEVICES count.")
    p.add_argument("--pipeline-parallel-size", type=int, default=1,
                   help="vLLM pipeline parallelism (multi-node; prefer TP within a node).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="vLLM per-GPU KV-cache memory fraction.")
    p.add_argument("--device-map", default="",
                   help="transformers backend only: 'auto' naive-shards across GPUs.")
    p.add_argument("--frames-mode", choices=["video", "extract"], default="video")
    p.add_argument("--frames-root", default="data/frames_test")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--temporal-num-frames", type=int, default=0,
                   help="Frame budget for time-sensitive tasks (temporal); "
                        "0 = same as --num-frames. Mirror the build value.")
    p.add_argument("--temporal-max-side", type=int, default=0,
                   help="Max side for temporal frames (0 = same as --max-side).")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--closed-scoring", choices=["logprob", "vote"], default="logprob",
                   help="How bcq/mcq decisions are made (method.md §9 W1). "
                        "'logprob': ONE deterministic pass per item reading the "
                        "first-token distribution over Yes/No / A-D (calibrated "
                        "margins for structural.py, no prose drift); mcq and "
                        "mcq_openended distributions are pooled via option-text "
                        "matching before choosing. 'vote': legacy n-sample "
                        "self-consistency voting (also persists vote counts).")
    p.add_argument("--no-conditioned-openended", action="store_true",
                   help="W1: by default mcq_openended explanations are generated "
                        "in a second pass conditioned on the pooled letter choice "
                        "(guarantees mcq<->mcq_oe consistency + on-topic content). "
                        "This flag falls back to independent greedy generation "
                        "with only the leading letter rewritten.")
    p.add_argument("--mcq-permute", type=int, default=0,
                   help="PROVE: score mcq/mcq_openended under N cyclic option-text "
                        "rotations and average the first-token distributions over "
                        "option text (position/letter-bias debias; 0/1 = off, 4 = "
                        "the full cycle). Only the max_tokens=1 scoring pass sees a "
                        "permuted question — generation prompts stay verbatim.")
    p.add_argument("--vote-n", type=int, default=5,
                   help="Self-consistency samples for bcq/mcq (closed-scoring=vote).")
    p.add_argument("--vote-temp", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0, help="Debug: first N items only.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Only infer these task types (e.g. mcq bcq) — to cheaply "
                        "re-run a fixed subset and splice it into an existing run.")
    p.add_argument("--no-strict-format", action="store_true",
                   help="Write predictions even if any fail the post-enforcement "
                        "parseability check (default: abort — see _report_format).")
    return p.parse_args()


def load_test_items(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    out = []
    for it in items:
        out.append({
            "item_index": it.get("item_index") or it.get("index") or it.get("id"),
            "video_id": it.get("video_id", ""),
            "question": it.get("question", ""),
            "task": it.get("task_type") or it.get("task") or infer_task_key(it),
        })
    return out


def build_engine(args):
    """Construct an ms-swift engine for the base model + LoRA adapter.

    The installed ms-swift exposes engines from the top-level ``swift`` package
    (no ``swift.llm``); ``TransformersEngine`` is the non-vLLM backend. The engine
    auto-builds the correct Qwen3-VL template, and each request carries its own
    system prompt, so no manual template binding is needed.
    """
    from swift import BaseArguments, TransformersEngine, VllmEngine
    # temporal frames are passed as an image list, so the per-prompt image cap
    # must cover whichever budget is larger (base vs temporal).
    max_frames = max(args.num_frames, args.temporal_num_frames or 0)
    common = dict(max_model_len=args.max_model_len,
                  limit_mm_per_prompt={"image": max_frames + 1, "video": 2})
    if args.adapter:
        model = BaseArguments.from_pretrained(args.adapter).model  # base model path
        adapters = [args.adapter]
    else:
        model, adapters = args.model, None

    if args.backend == "vllm":
        # Multi-GPU: tensor_parallel_size shards one model across N GPUs (the caller
        # sets it from CUDA_VISIBLE_DEVICES). getattr keeps callers whose namespace
        # predates these fields working (defaults => single GPU, unchanged behavior).
        engine = VllmEngine(
            model, enable_lora=bool(adapters), max_loras=1,
            max_lora_rank=args.max_lora_rank,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1) or 1,
            pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1) or 1,
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9) or 0.9,
            **common)
    else:
        # transformers backend: device_map='auto' naive-shards the model across GPUs.
        engine = TransformersEngine(
            model, adapters=adapters, attn_impl=args.attn_impl,
            device_map=(getattr(args, "device_map", "") or None))
    return engine, adapters


def _video_field(args, item, spec, frame_cache):
    """Return (videos_list, video_hint) for the request, per task frame plan.

    Temporal items get denser timestamped extract frames even when the global
    mode is ``video`` (mirrors build_dataset; see method.md §3.3).
    """
    vid = item["video_id"]
    vpath = vid if os.path.isabs(vid) else os.path.join(args.videos_root, vid)
    eff_mode, eff_frames, eff_side = frame_plan(
        spec, args.frames_mode, args.num_frames, args.max_side,
        args.temporal_num_frames or None, args.temporal_max_side or None)
    if eff_mode == "extract":
        ck = (vid, eff_frames, eff_side)
        if ck not in frame_cache:
            out_dir = os.path.join(
                args.frames_root, f"{frame_utils.safe_name(vid)}_n{eff_frames}_s{eff_side}")
            frame_cache[ck] = frame_utils.extract_uniform_frames(
                vpath, out_dir, eff_frames, eff_side)
        fr = frame_cache[ck]
        hint = frame_utils.timestamp_hint(fr.timestamps, fr.duration) \
            if spec.metric == "iou" else ""
        return [fr.paths], hint
    # video mode: probe duration for the temporal task's hint. Uses the *same*
    # shared helper as build_dataset so train/test temporal framing never diverges.
    hint = ""
    if spec.metric == "iou" and os.path.exists(vpath):
        hint = frame_utils.duration_hint(frame_utils.video_duration(vpath))
    return [vpath], hint


# ---------------------------------------------------------------------------
# W1 — first-token logprob scoring for the closed-family tasks (method.md §9)
# ---------------------------------------------------------------------------

def _supports_logprobs(request_config_cls) -> bool:
    import dataclasses as dc
    try:
        names = {f.name for f in dc.fields(request_config_cls)}
    except TypeError:
        return False
    return "logprobs" in names and "top_logprobs" in names


def _canon_first_token(token: str, kind: str):
    """Map a raw top-logprob token to its canonical decision token, else None."""
    t = (token or "").strip()
    if kind == "yesno":
        low = t.lower()
        if low in ("yes", "no"):
            return low.capitalize()
        return None
    up = t.upper()
    return up if up in ("A", "B", "C", "D") else None


def _first_token_dist(choice, kind: str) -> dict:
    """{canonical token: probability} from a choice's first-token top-logprobs.

    Tolerates both dict- and object-shaped logprobs payloads (vLLM vs
    transformers backends serialize them differently). Returns {} when the
    backend produced no usable logprobs, so callers can fall back gracefully.
    """
    import math

    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    lp = _get(choice, "logprobs")
    content = _get(lp, "content") if lp is not None else None
    if not content:
        return {}
    tops = _get(content[0], "top_logprobs") or []
    probs: dict = {}
    for t in tops:
        canon = _canon_first_token(_get(t, "token") or "", kind)
        logprob = _get(t, "logprob")
        if canon is not None and logprob is not None and canon not in probs:
            probs[canon] = math.exp(float(logprob))
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()} if total > 0 else {}


_LEAD_TOKEN = re.compile(r"^(yes|no|[A-D])\b[.,:)]?\s*", re.IGNORECASE)


def _retoken(text: str, token: str) -> str:
    """Force an answer to open with ``token`` (keep the rest of the text)."""
    text = (text or "").strip()
    m = _LEAD_TOKEN.match(text)
    rest = text[m.end():] if m else text
    return f"{token}. {rest}".strip()


def _pool_mcq_letters(metas: list) -> dict:
    """Final letter per mcq/mcq_openended item, pooling twin distributions.

    The two tasks carry the same option set on most test videos but with the
    letters SHUFFLED (data.md §5.3), so distributions are pooled over the
    normalized option *text* and the winner is re-expressed in each item's own
    letters. Items without a scored twin fall back to their own argmax.
    Reads the per-item first-token distribution from ``meta['votes']``.
    """
    by_video: dict = {}
    for i, m in enumerate(metas):
        if m["task"] in ("mcq", "mcq_openended") and m.get("votes"):
            by_video.setdefault(m["video_id"], {}).setdefault(m["task"], []).append(i)

    final: dict = {}
    for vid, tasks in by_video.items():
        mcqs = tasks.get("mcq", [])
        opens = {question_stem(metas[j]["question"]): j
                 for j in tasks.get("mcq_openended", [])}
        paired = set()
        for i in mcqs:
            j = opens.get(question_stem(metas[i]["question"]))
            if j is None:
                continue
            opts_i = question_options(metas[i]["question"])
            opts_j = question_options(metas[j]["question"])
            score = {}
            for L, p in metas[i]["votes"].items():
                if L in opts_i:
                    score[opts_i[L]] = score.get(opts_i[L], 0.0) + p
            for L, p in metas[j]["votes"].items():
                if L in opts_j:
                    score[opts_j[L]] = score.get(opts_j[L], 0.0) + p
            if not score:
                continue
            best_text = max(score, key=score.get)
            back_i = {v: k for k, v in opts_i.items()}
            back_j = {v: k for k, v in opts_j.items()}
            if best_text in back_i and best_text in back_j:
                final[i], final[j] = back_i[best_text], back_j[best_text]
                paired.update((i, j))
        for i in mcqs + list(opens.values()):
            if i not in paired:
                final[i] = max(metas[i]["votes"], key=metas[i]["votes"].get)
    return final


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    items = load_test_items(args.test_json)
    if args.tasks:
        keep = set(args.tasks)
        items = [it for it in items if it["task"] in keep]
    if args.limit:
        items = items[: args.limit]

    from swift import InferRequest, RequestConfig
    engine, adapters = build_engine(args)
    adapter_request = None
    if adapters and args.backend == "vllm":
        from swift import AdapterRequest
        adapter_request = AdapterRequest("track3", adapters[0])

    frame_cache: dict = {}
    requests, metas = [], []
    for it in items:
        spec = get_task(it["task"])
        videos, hint = _video_field(args, it, spec, frame_cache)
        user = build_user_prompt(spec, it["question"], hint)
        messages = [{"role": "system", "content": spec.system},
                    {"role": "user", "content": user}]
        requests.append(InferRequest(messages=messages, videos=videos))
        metas.append(it)

    extra = {"adapter_request": adapter_request} if adapter_request else {}
    greedy = RequestConfig(max_tokens=args.max_new_tokens, temperature=0)

    closed_logprob = args.closed_scoring == "logprob"
    if closed_logprob and not _supports_logprobs(RequestConfig):
        print("[infer][WARN] installed ms-swift RequestConfig has no logprobs/"
              "top_logprobs — falling back to --closed-scoring vote.")
        closed_logprob = False
    # Conditioned mcq_oe = the SFT prompt UNCHANGED + the pooled letter forced as
    # a response_prefix ("B. "), so the explanation continues in-distribution.
    # MEASURED (leaderboard v5-2): appending an instruction to the prompt instead
    # drifted the answer register and cost mcq_oe −0.13 BERTScore — never modify
    # the prompt text for this pass. Needs per-request chat_template_kwargs.
    import dataclasses as _dc
    supports_prefix = any(f.name == "chat_template_kwargs"
                          for f in _dc.fields(InferRequest))
    conditioned_oe = (closed_logprob and not args.no_conditioned_openended
                      and supports_prefix)
    if closed_logprob and not args.no_conditioned_openended and not supports_prefix:
        print("[infer][WARN] installed ms-swift InferRequest has no "
              "chat_template_kwargs (response_prefix unsupported) — mcq_openended "
              "falls back to greedy generation + leading-letter rewrite.")

    # Stage A — one greedy text pass for every item that needs generated text.
    # Strip <think>...</think> reasoning here so the raw predictions jsonl (and the
    # submission built from it) carry only the answer. Under logprob scoring,
    # bcq/mcq need no text (the decision IS the first token), and mcq_openended is
    # generated in stage C conditioned on the pooled letter.
    skip_text = ({"bcq", "mcq"} | ({"mcq_openended"} if conditioned_oe else set())) \
        if closed_logprob else set()
    preds = [""] * len(metas)
    gen_idx = [i for i, m in enumerate(metas) if m["task"] not in skip_text]
    if gen_idx:
        resp = engine.infer([requests[i] for i in gen_idx], greedy, **extra)
        for j, i in enumerate(gen_idx):
            preds[i] = strip_reasoning(resp[j].choices[0].message.content)

    if closed_logprob:
        # Stage B — first-token distribution for the closed family (W1). One
        # deterministic max_tokens=1 request per item; the Yes/No / A-D
        # probabilities are the decision AND the margins structural.py consumes.
        score_idx = [i for i, m in enumerate(metas) if m["task"] in CLOSED_TOKEN_KIND]
        if score_idx:
            # --mcq-permute R: additionally score the letter tasks under R-1 cyclic
            # rotations of the option TEXTS and average the distributions over
            # option text (track3.mcq_debias — position/letter-bias removal). Only
            # this max_tokens=1 pass sees a permuted question; a malformed question
            # falls back to the single verbatim pass.
            jobs = []   # (item_i, slot->original-letter mapping | None, request)
            for i in score_idx:
                jobs.append((i, None, requests[i]))
                if args.mcq_permute > 1 and CLOSED_TOKEN_KIND[metas[i]["task"]] == "letter":
                    spec = get_task(metas[i]["task"])
                    for s in range(1, args.mcq_permute):
                        perm = permuted_question(metas[i]["question"], s)
                        if perm is None:
                            break
                        pq, mapping = perm
                        jobs.append((i, mapping, InferRequest(
                            messages=[{"role": "system", "content": spec.system},
                                      {"role": "user",
                                       "content": build_user_prompt(spec, pq, "")}],
                            videos=requests[i].videos)))
            cfg = RequestConfig(max_tokens=1, temperature=0,
                                logprobs=True, top_logprobs=20)
            resp = engine.infer([r for _, _, r in jobs], cfg, **extra)
            by_item: dict[int, list] = {}
            for (i, mapping, _), r in zip(jobs, resp):
                dist = _first_token_dist(r.choices[0],
                                         CLOSED_TOKEN_KIND[metas[i]["task"]])
                by_item.setdefault(i, []).append((dist, mapping))
            ident = {L: L for L in "ABCD"}
            for i, pairs in by_item.items():
                if len(pairs) == 1:
                    dist = pairs[0][0]
                else:
                    dist = merge_permuted_votes([d for d, _ in pairs],
                                                [m or ident for _, m in pairs])
                if dist:
                    metas[i]["votes"] = dist
            unscored = [i for i in score_idx if "votes" not in metas[i]]
            if unscored:
                # Backend returned no usable logprobs — regenerate text for the
                # items whose prediction would otherwise be empty (bcq/mcq/mcq_oe).
                print(f"[infer][WARN] no first-token logprobs for {len(unscored)} "
                      f"item(s); falling back to greedy text for those.")
                need = [i for i in unscored if i not in gen_idx]
                if need:
                    resp = engine.infer([requests[i] for i in need], greedy, **extra)
                    for j, i in enumerate(need):
                        preds[i] = strip_reasoning(resp[j].choices[0].message.content)

        # bcq: the scored token is the prediction; bcq_openended: keep the
        # generated explanation but force it to open with the scored token.
        for i, m in enumerate(metas):
            dist = m.get("votes")
            if not dist:
                continue
            top = max(dist, key=dist.get)
            if m["task"] == "bcq":
                preds[i] = top
            elif m["task"] == "bcq_openended":
                preds[i] = _retoken(preds[i], top)

        # mcq cluster: pool twin distributions over option text, then choose.
        final_letters = _pool_mcq_letters(metas)
        for i, letter in final_letters.items():
            if metas[i]["task"] == "mcq":
                preds[i] = letter

        # Stage C — mcq_openended explanations conditioned on the pooled letter:
        # the original SFT prompt verbatim, with the letter forced via
        # response_prefix so the model *continues* "B. ..." in its trained
        # register (decode prepends the prefix, so content arrives as "B. ...").
        cond_idx = [i for i in final_letters if metas[i]["task"] == "mcq_openended"]
        if conditioned_oe and cond_idx:
            cond_reqs = [InferRequest(
                messages=requests[i].messages, videos=requests[i].videos,
                chat_template_kwargs={"response_prefix": f"{final_letters[i]}. "})
                for i in cond_idx]
            resp = engine.infer(cond_reqs, greedy, **extra)
            for j, i in enumerate(cond_idx):
                text = strip_reasoning(resp[j].choices[0].message.content)
                preds[i] = _retoken(text, final_letters[i])
        else:
            # No conditioned pass: keep the in-distribution greedy explanation,
            # only rewrite the leading letter (measured safe: bcq_oe −0.003).
            for i in cond_idx:
                preds[i] = _retoken(preds[i], final_letters[i])
    elif args.vote_n > 1:
        # Legacy self-consistency voting (persists vote counts for structural.py).
        vote_idx = [i for i, m in enumerate(metas)
                    if get_task(m["task"]).metric in (METRIC_ACC_YESNO, METRIC_ACC_LETTER)]
        if vote_idx:
            preds = _vote(engine, requests, metas, preds, vote_idx, args, extra, InferRequest, RequestConfig)

    # 3) STRICT FORMAT ENFORCEMENT — the single choke-point. Coerce every output
    #    to the exact string the official grader parses, so the predictions jsonl
    #    is already submission-ready (scoring it raw == scoring the CSV, and
    #    temporal is fenced JSON not prose). The pre-enforcement text (reasoning
    #    already stripped / vote already taken) is kept under "raw_pred" for audit;
    #    make_submission / eval_local re-apply submission() idempotently.
    bad: dict[str, list] = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for it, raw_pred in zip(metas, preds):
            spec = get_task(it["task"])
            pred = spec.submission(raw_pred)
            if not check_parseable(spec, pred):
                bad.setdefault(it["task"], []).append(it["item_index"])
            rec = {"item_index": it["item_index"], "video_id": it["video_id"],
                   "task": it["task"], "prediction": pred, "raw_pred": raw_pred}
            if it.get("votes"):
                rec["votes"] = it["votes"]  # margins for track3.structural rules 4.2/4.3
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(metas)} predictions to {args.out}")
    _report_format(metas, bad, strict=not args.no_strict_format)


def _report_format(metas, bad: dict, strict: bool) -> None:
    """Per-task parseability report over the enforced predictions. Enforcement
    guarantees 0 bad; this is a regression tripwire so a future extractor/grader
    drift can never ship silently."""
    by_task: dict[str, int] = {}
    for it in metas:
        by_task[it["task"]] = by_task.get(it["task"], 0) + 1
    print("Format check (grader-parseable / total):")
    for task in sorted(by_task):
        n = by_task[task]
        nbad = len(bad.get(task, []))
        marker = "ok  " if nbad == 0 else "BAD "
        print(f"  {marker} {task:24} {n - nbad}/{n}")
    total_bad = sum(len(v) for v in bad.values())
    if total_bad and strict:
        raise SystemExit(
            f"[infer] {total_bad} prediction(s) are not grader-parseable after "
            f"enforcement: {bad}. This should be impossible — investigate "
            f"tasks.enforce_format / the official extractors. Pass "
            f"--no-strict-format to write anyway.")


def _vote(engine, requests, metas, preds, vote_idx, args, extra, InferRequest, RequestConfig):
    sub = [requests[i] for i in vote_idx]
    cfg = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.vote_temp,
                        n=args.vote_n)
    resp = engine.infer(sub, cfg, **extra)
    for j, i in enumerate(vote_idx):
        spec = get_task(metas[i]["task"])
        votes = Counter(vote_token(spec, c.message.content) for c in resp[j].choices)
        metas[i]["votes"] = dict(votes)  # margins for track3.structural rules 4.2/4.3
        preds[i] = votes.most_common(1)[0][0]
    return preds


if __name__ == "__main__":
    main()
