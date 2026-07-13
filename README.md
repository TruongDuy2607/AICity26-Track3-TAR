<div align="center">

# AI City Challenge 2026 — Track 3: Traffic Anomaly Reasoning (TAR)

**Team 12 — a unified Qwen3-VL-32B backbone + a verified, metric-aligned post-processing chain**

</div>

One LoRA-tuned **Qwen3-VL-32B-Instruct** backbone (no per-task models) followed by a
deterministic, partial-safe post-processing chain. Two scripts reproduce everything:
`phase0_base_sft.sh` (train) → `phase3_infer.sh` (infer → submission). The chain lifts
the leaderboard mean (unweighted mean of the 9 graded tasks) from **0.5930** (base SFT) to **0.6669**.

---

## 1. Setup

### a. Environment
> **Anaconda**, per [requirements.txt](requirements.txt) (CUDA 12.8 /
vLLM 0.11 / ms-swift / transformers 4.57):

```bash
conda create -n vlm python=3.12 -y && conda activate vlm
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip install ms-swift -U
pip install transformers==4.57.0
conda install -c nvidia cuda-nvcc=12.8.93 -y
pip install flash-attn --no-build-isolation
pip install qwen-vl-utils==0.0.14 deepspeed decord wandb bert-score
```

### b. Data
`bash scripts/download_data.sh` (annotations + test clips), then place the
8 upstream train-video sources under `train/videos/` (see the HF README). Expected
layout under `$TAR_ROOT`:

```
$TAR_ROOT/
├── train/   # ANN_DIR
│   ├── bcq.json  mcq.json  bcq_openended.json  mcq_openended.json  open_qa.json
│   ├── causal_linkage.json  scene_description.json  temporal_description.json
│   ├── temporal_localization.json  video_summarization.json
│   └──  videos/<sub-dataset>    # TRAIN_VIDEOS_ROOT
│   
└── test/
    ├── test.json
    ├── clip_manifest.csv
    ├── download_test_videos.py
    ├── evaluate.py
    └── videos/<video_id>        # TEST_VIDEOS_ROOT
```

Train = 3,669 videos × 10 tasks → 44,040 items (auto-labelled). Test = 960 items / 80
clips (human-curated). Pipeline outputs live outside `$TAR_ROOT` in `data/processed/`,
`output/`, `preds/`, `submissions/`.

### c. Values to edit
All machine-specific values live in **[init.sh](init.sh)**;
edit it once, then `source init.sh` before every run. It overrides the defaults in
`configs/common.sh` + `configs/qwen3vl_32b_lora.sh` (which stay untouched):

| Value | What |
|---|---|
| `TAR_ROOT` | dataset root (layout above) |
| `MODEL` | Qwen3-VL-32B-Instruct dir or HF id |
| `CUDA_VISIBLE_DEVICES` | GPUs (default profile assumes 4× A100-80G) |
| `HF_TOKEN`, `WANDB_API_KEY` | optional secrets (empty ⇒ skipped; wandb off with `REPORT_TO=tensorboard`) |

Activate the conda env yourself (`conda activate vlm`) before `source init.sh` — the
scripts run in whatever env is active.

Defaults are already `MODEL_CONFIG=qwen3vl_32b_lora` and `PROFILE=a100_80g_4x_32b`
(4× A100-80G, DeepSpeed ZeRO-3); the LoRA/optim recipe (r16/α32 `all-linear`, ViT +
aligner frozen, 2 epochs, lr 2e-4, bf16, 32 frames) is baked in — do not change it to
reproduce.

---

## 2. Run

### Step-1: Prepare Enviroments and Training data

```bash
conda activate vlm 
source init.sh
```

### Step-2: Train — base multi-task LoRA SFT
```bash
# Train — base multi-task LoRA SFT (scripts/prove/phase0_base_sft.sh).
bash scripts/prove/phase0_base_sft.sh
```
Step-2 already merges its last checkpoint. To merge a **different** checkpoint by hand
(the exact `swift export` it runs internally) and get a `MODEL_PATH`:

```bash
swift export --adapters output/<run>/checkpoint-XXXX --merge_lora true 
```

### Step-3: Inference
```bash
# (2) Infer — full inference chain -> FINAL submission (scripts/prove/phase3_infer.sh).
#     MODEL_PATH is required: point it at the merged dir from step (1).
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" GPU_MEM_UTIL=0.85 NUM_FRAMES=32 \
MODEL_PATH="$(cat output/prove/BASE_MODEL_PATH)" \
    bash scripts/prove/phase3_infer.sh
```


Phase 3 runs 9 toggleable, partial-safe stages (each `DO_X=0` skips; a skipped stage
degrades to the previous behaviour):

| # | Stage | Output |
|---|---|---|
| 1 | infer: W1 first-token logprob scoring + MCQ permutation debias | `preds/test_pred.jsonl` |
| 2 | structural pre-pass (`--all`: BCQ pairing, temporal prior) | `preds/*.struct0.jsonl` |
| 3 | scene-attribute probes | `preds/scene_probes.jsonl` |
| 4 | Evidence-Sheet renders + K-candidate pool | `preds/text_dossier.jsonl`, `candidates.jsonl` |
| 5 | claim-level verification (W1 probes) | `preds/verify.jsonl` |
| 6 | verify-weighted MBR selection + TD untrim | `preds/text_override.untrim.jsonl` |
| 7 | MCQ-OE anchor, merged over the override | `preds/text_override.final.jsonl` |
| 8 | structural + submission CSV + validation | `submissions/submission-prove.csv` |
| 9 | td700 TD re-render (budget 0.8, K16) + MBR, overlaid + re-post | **`submissions/submission-prove-td700.csv`** (FINAL) |

Key knobs (sane defaults): `MCQ_PERMUTE=4`, `N_SAMPLES=8`, `SAMPLE_TEMPERATURE=0.8`,
`LAM=0.3`, `DO_TD700=1`, `TD700_LENGTH_TOL=0.8`, `TD700_N_SAMPLES=16`,
`PROBE_BATCH/GEN_BATCH=256` (host-RAM guard — lower if the pod is OOM-Killed). The
validator must report **960/960** rows parseable.

---

## 3. Results (cumulative ablation, [method-offiicial.md](method-offiicial.md) §4)

Mean = unweighted mean of the 9 graded tasks (Temporal mIoU excluded).

| Stage | Added component | Mean |
|---|---|---|
| p0 | Base SFT + W1 logprob + MCQ debias | 0.5930 |
| p1 | + Structural rules (BCQ pairing, temporal prior) | 0.6047 |
| p2 | + Evidence-Sheet renders | 0.6393 |
| p3 | + Verify-weighted MBR (λ=0.3) | 0.6485 |
| p4 | + TD untrim | 0.6540 |
| p5 | + MCQ-OE anchor | 0.6657 |
| p6 | + td700 | **0.6669** |

---

## 4. Notes

- **vLLM + ViT-LoRA:** vLLM cannot apply Qwen3-VL ViT-LoRA from an `all-linear`
  adapter → Phase 0 auto-merges; always pass a **merged** dir (or `BACKEND=transformers`).
- **Frames must match training** (32); a mismatch hurts narrative *and* temporal.
- **Temporal column is override-bound** (structural prior injects the leaked question
  window) and excluded from the mean.
- **Local eval is a tripwire only** — test answers are human-curated, so held-out-train
  BERTScore absolutes are not leaderboard-faithful; gate on deltas.
- **Compliance:** every override reads only the released test *questions* + the model's
  own outputs — single unified checkpoint, no test annotations.

---

## 5. License

Released for reproducibility under the AI City Challenge 2026 award requirements (set an
OSS license in `LICENSE` before public release). The official scorer `evaluate.py` and
the TAR dataset / its upstream sources retain their original terms; test clips are cut
from public YouTube videos under the dataset's research terms.
