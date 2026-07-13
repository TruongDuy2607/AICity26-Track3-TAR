#!/usr/bin/env bash
# Hardware profile: 2 x A100 40GB (DDP, halve grad-accum to keep effective batch ~16).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE=2
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export DEEPSPEED="${DEEPSPEED:-zero2}"        # use zero3 if you raise batch/seq-len
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
