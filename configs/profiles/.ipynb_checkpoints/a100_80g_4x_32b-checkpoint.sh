#!/usr/bin/env bash
# Hardware profile: 4 x A100 40GB.
#
# Calibrated from the proven OOM-safe 2x run (per_device_batch=2, grad_accum=4,
# effective batch = 2 * 4 * 2 = 16 at NUM_FRAMES=8, DeepSpeed ZeRO-2).
#
# Two changes vs 2x, both to keep VRAM in budget while we ~DOUBLE the vision-token
# load (Stream-1 raised NUM_FRAMES 8->16 and TEMPORAL_NUM_FRAMES->24; more frames =
# more activations per sample):
#   1) DeepSpeed ZeRO-3 (was ZeRO-2). With LoRA the 8B base is frozen, but ZeRO-2
#      still *replicates* its ~16GB of weights on every GPU. ZeRO-3 shards those
#      frozen params across all 4 ranks (~4GB/GPU), freeing ~12GB/GPU for the larger
#      frame activations. This is the main "use VRAM optimally" lever at 4x.
#   2) grad_accum 4->2 so effective batch stays ~16 (2 * 2 * 4 = 16) now that there
#      are 4 data-parallel ranks instead of 2 — same recipe, ~2x faster.
#
# Keep per_device_batch=2 (the measured OOM-safe value). If the bigger frame budget
# still OOMs: drop to PER_DEVICE_TRAIN_BATCH_SIZE=1 GRAD_ACCUM=4 (effective 16), or
# lower NUM_FRAMES. To push a *larger* effective batch (~32) instead, set GRAD_ACCUM=4.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"          # effective batch = 2 * 2 * 4 = 16
# ZeRO-3 to shard the frozen 8B base across 4 GPUs. We do NOT use the builtin
# "zero3" preset: its stage3_param_persistence_threshold="auto" makes DeepSpeed
# coalesce the model's small "persistent" params into one flat all-gather buffer,
# and Qwen3-VL mixes fp32 (norm/rotary/visual-merger) with bf16 there -> the
# optimizer step dies with `output tensor must have the same type as input tensor`
# (deepspeed stage3._post_step -> _allgather_params_coalesced). Our config pins
# threshold=0 (no persistent params -> that gather path is skipped) + bf16. ZeRO-2
# never hit this (it replicates frozen params, no coalesced gather).
# Fallback if anything else misbehaves: DEEPSPEED=zero2 (the proven 2x recipe).
_PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEEPSPEED="${DEEPSPEED:-$_PROFILE_DIR/../ds/zero3_vlm.json}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
