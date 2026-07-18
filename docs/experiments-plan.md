# EDCR — Kế hoạch thí nghiệm bổ sung cho paper

> Mục tiêu: chuyển paper từ *"list các stage + số leaderboard 0.6669"* sang *"một cơ chế
> dựa trên tính chất của model (verification–generation gap), được kiểm chứng có kiểm soát"*.
> Reviewer cần biết **vì sao** EDCR hoạt động, không chỉ **kết quả**. Trọng tâm: làm nổi bật
> và chứng minh tính hiệu quả + tính general của **Cross-Question Evidence Sheet**.
>
> **✅ ĐÃ TRIỂN KHAI** → `scripts/ablations/` (xem `scripts/ablations/README.md` cho usage
> tiếng Anh). E1–E5 đều chạy được trên checkpoint base + `val_gt.json`, đồng nhất setup với
> test (chỉ đổi split + đúng 1 biến/lần). Chạy: `bash scripts/ablations/00_base_val.sh` rồi
> `bash scripts/ablations/run_all.sh`.

## 0. Bối cảnh & định vị

- **Leaderboard đã đóng.** Con số cuối cùng (test, 80 clip human-curated) = **0.6669**, giữ nguyên.
- **Tài sản chạy được:**
  - Checkpoint base SFT đã merge từ `scripts/prove/phase0_base_sft.sh`
    (đường dẫn ghi ở `output/prove/BASE_MODEL_PATH`).
  - `val.jsonl` + `val_gt.json`: `VAL_RATIO=0.02` tách **theo video** (`build_dataset.py:218`)
    → ~73/3670 video held-out, ~870 item, phủ đủ 10 task type. **Quy mô tương đương tập test.**
  - `scripts/eval.sh`: chạy infer trên `val_gt.json` (video resolve dưới `TRAIN_VIDEOS_ROOT`)
    rồi chấm bằng **grader chính thức** (`track3.eval_local` → `track3.official`). Số in ra =
    số leaderboard sẽ cho trên split này.
  - Toàn bộ module cơ chế: `text_dossier`, `claim_verify`, `mbr_select`, `structural`,
    và bộ corruption có sẵn (`evidence.corrupt_scene`, `evidence.jitter_evidence`,
    `build_render_sft --p-obs-flip/--p-scene/--p-obs-drop`).

### ⚠️ Định vị val — **mechanism analysis, không phải test proxy**

`val` references là **pseudo-label kiểu training** (máy sinh, văn phong CoT), KHÁC văn phong
human-curated của test. Vì vậy:

- **KHÔNG** dùng số tuyệt đối BERTScore trên val để tuyên bố ngang test.
- **CÓ** dùng val để đo: (a) *delta tương đối* giữa các điều kiện có kiểm soát,
  (b) *hành vi* (override rate, calibration, flip rate), (c) *thứ hạng* giữa các biến thể.

**Framing trong paper:** thêm mục **"§5.2 Why it works: a controlled analysis on a held-out
split"** ngay trước bảng cumulative (Table 2). Ghi rõ một câu:
> "All analyses in this section are run on a held-out split of the *training* videos with
> machine-generated references; we report them for relative deltas and mechanism behavior, not
> as estimates of the human-curated test score, which is reported in Table 2."

Điều này biến điểm yếu (val ≠ test) thành điểm mạnh: bạn **tách bạch** "vì sao nó hoạt động"
(val, general, mechanism) khỏi "kết quả cuối" (test, leaderboard) — đúng tinh thần *general hơn*.

---

## 1. Setup chung (chạy 1 lần)

```bash
# (a) build split (val_gt.json + val.jsonl). SKIP_BASE nếu đã có.
bash scripts/prepare_data.sh          # -> data/processed/{train,val}.jsonl, val_gt.json

# (b) đường dẫn checkpoint base
export MODEL_PATH="$(cat output/prove/BASE_MODEL_PATH)"
export VAL_GT=data/processed/val_gt.json
export VAL_VIDEOS=$TRAIN_VIDEOS_ROOT   # val video nằm dưới train root
```

**Chạy full chain trên val** (thay vì test) — trỏ `TEST_JSON`/`TEST_VIDEOS_ROOT` sang val:

```bash
TEST_JSON="$VAL_GT" TEST_VIDEOS_ROOT="$VAL_VIDEOS" \
  MODEL_PATH="$MODEL_PATH" bash scripts/prove/phase3_infer.sh
# rồi chấm submission bằng grader chính thức:
python -m track3.eval_local --gt "$VAL_GT" --submission submissions/submission-prove-td700.csv
```

> Mỗi thí nghiệm dưới đây tái dùng harness này, chỉ bật/tắt stage hoặc thay nguồn sheet.

---

## E1 — Định lượng verification–generation gap  ⭐ BẮT BUỘC

**Vá lỗ hổng #1:** section `sec:gap` đang bị comment out — luận điểm trung tâm (`P_θ` calibrated,
`G_θ` không) hiện **không có số đo nào**.

**Giả thuyết:** cùng một fact về cùng một clip, model trả lời **đúng như một câu yes/no**
(`P_θ`) nhưng **nói sai khi nhúng trong đoạn văn** (`G_θ`).

### E1a — Gap accuracy (bảng số chính)

1. **Nhánh verifier:** BCQ/MCQ trên val qua `P_θ` (Eq.3–6). Chấm accuracy.
   ```bash
   MODEL_PATH="$MODEL_PATH" bash scripts/eval.sh   # in per-type acc, gồm BCQ/MCQ
   ```
2. **Nhánh generator trên CHÍNH fact đó:** sinh narrative KHÔNG sheet (clip-only), rồi trích lại
   cùng fact từ đoạn văn bằng `claim_verify.decompose_claims` + probe, so với GT.
   - Chọn fact có mặt cả ở dạng closed-form *và* nhúng trong narrative: `agent identity`,
     `có/không va chạm`, `root cause`.
3. **Bảng cần in:**

   | Fact | Verifier acc (`P_θ`) | Generator acc (`G_θ` narrative) | Gap |
   |---|---|---|---|
   | Collision present (BCQ) | ~0.9–1.0 | ? | Δ |
   | Agent identity | ~0.9 | ? | Δ |
   | Root cause (MCQ) | ~0.9 | ? | Δ |

   > "The asymmetry is not that the model cannot see the event; it is that it cannot reliably
   > *say* what it saw." → biện minh trực tiếp Eq.(crossq).

### E1b — Reliability diagram (1 figure)

Bin theo margin `m_q` (Eq.6) → accuracy vs margin bin + **ECE**. Chứng minh margin là tín hiệu
tin cậy ⇒ biện minh **toàn bộ safe-fallback gate γ** và câu "near-saturated" (§4.5).

- Cần: dump `m_q` per item từ `infer` (logprob path). Kiểm tra `track3/infer.py` có ghi margin
  vào pred jsonl chưa; nếu chưa → thêm field `margin` (nhỏ, ~10 dòng).
- Figure: reliability curve (perfect-calibration diagonal + observed) cho BCQ và MCQ.

**Cần code:** (1) script trích fact từ narrative để so GT (tái dùng `claim_verify`); (2) expose
`margin` trong pred jsonl nếu chưa có; (3) script vẽ reliability. Ước lượng: nửa ngày.

**Viết vào paper:** khôi phục `sec:gap` với 1 bảng (E1a) + 1 figure (E1b).

---

## E2 — Ablation cô lập Cross-Question Evidence Sheet  ⭐ BẮT BUỘC

**Vấn đề:** Table 2 là *cumulative* → gain của sheet (+0.0346) bị confound bởi thứ tự stage.
Reviewer cần A/B **cô lập sheet, giữ mọi thứ khác cố định** (greedy, KHÔNG MBR, KHÔNG length-recovery).

### E2a — Bảng A/B chính (4 điều kiện)

| ID | Điều kiện | Nguồn sheet | Kỳ vọng | Ý nghĩa |
|---|---|---|---|---|
| B0 | baseline | KHÔNG sheet (clip-only) | thấp nhất | mốc gốc |
| B1 | +sheet (ours) | probe `P_θ` confident | > B0 | gain sạch kênh evidence |
| B2 | oracle | **GT facts** | trần trên | trần của kênh evidence |
| B3 | wrong sheet | facts lấy từ **clip KHÁC** (sai, đúng format/độ dài) | **< B0** ⇒ then chốt | bác bỏ "chỉ là prompt dài hơn" |

Cách chạy (greedy, tắt các stage sau):

```bash
# B1 (ours): dossier greedy, KHÔNG MBR/verify/length-recovery
DO_VERIFY=0 DO_MBR=0 DO_UNTRIM=0 DO_TD700=0 DO_ANCHOR=0 DO_STRUCT0=1 \
  N_SAMPLES=1 TEST_JSON="$VAL_GT" TEST_VIDEOS_ROOT="$VAL_VIDEOS" \
  MODEL_PATH="$MODEL_PATH" bash scripts/prove/phase3_infer.sh
python -m track3.eval_local --gt "$VAL_GT" --submission submissions/submission-prove.csv

# B0: cùng lệnh nhưng sheet rỗng  -> cần flag --no-sheet (xem "Cần code")
# B2: --fact-policy trust + nạp GT facts (build_render_sft đã dựng sheet từ GT)
# B3: permute facts giữa các clip trước khi render
```

### E2b — Leave-one-field-out (bảng đã bị cắt khỏi paper, reviewer sẽ đòi)

Drop lần lượt từng field {`RootCause`, `Agents`, `Scene`, `Consequence`, `Observations`} khỏi sheet,
đo delta per-type. Cho biết *fact nào* mang gain. Tái dùng `evidence.jitter_evidence` (hiện drop
ngẫu nhiên 1 field) → mở rộng thành drop **cố định theo tên field**.

| Field bị drop | ΔQA | ΔCL | ΔSD | ΔTD | ΔMean |
|---|---|---|---|---|---|
| (none = B1) | — | — | — | — | 0 |
| −RootCause | ? | ? | ? | ? | ? |
| −Agents | … | | | | |

**Giá trị:** (1) B0→B1 = gain sạch, không confound. (2) B1 vs B2 **phân rã** gain thành
"giá trị kênh evidence" (B2−B0) và "chất lượng probe `P_θ`" (B1/B2). (3) **B3 < B0** là bằng
chứng generality mạnh nhất: sheet có tác dụng nhờ *nội dung fact*, không phải scaffolding/độ dài.

**Cần code:** (1) `text_dossier --no-sheet` (render clip-only, ~15 dòng — sheet rỗng ⇒
`render_evidence_prompt` bỏ hết dòng fact). (2) helper permute facts across clips cho B3
(~30 dòng, tái dùng `EvidenceSheet`). (3) drop-by-name trong `jitter_evidence` (~10 dòng).
Ước lượng: 1 ngày (gồm chạy).

**Viết vào paper:** thay/bổ sung 1 bảng cô lập (E2a) + 1 bảng leave-one-out (E2b) trong §5.2.

---

## E3 — Chiều của luồng thông tin (P→G vs G→G)  — Cao

**Bảo vệ thesis:** *"every mechanism moves information from the verifier to the generator, and
none the other way."* Skeptic hỏi: nếu chỉ feed lại một draft narrative thì sao?

| ID | Sheet lắp từ | Kỳ vọng |
|---|---|---|
| P→G (ours) | closed-form probe `P_θ` (calibrated) | cao |
| G→G | fact do **narrative pass đầu của `G_θ`** sinh ra (không qua probe) | thấp hơn |

**Kết luận cần:** P→G > G→G ⇒ chính **kênh calibrated** tạo gain, không phải "conditioning bất kỳ".
Đây là lập luận làm paper *general* (nói về asymmetry, không về benchmark).

**Cần code:** một nguồn sheet mới "self-conditioning": chạy `G_θ` pass 1 (không sheet) → trích
fact bằng `claim_verify` → verbalise thành sheet → pass 2. Ước lượng: 1 ngày.

**Viết vào paper:** 1 dòng bảng trong §5.2 hoặc trong phần negative/ablation.

---

## E4 — Định lượng hành vi override  — Cao (vá placeholder `X/80`)

**Vá §6 + Fig.2:** sheet cố ý chứa fact sai + instruction "correct any conflicting fact".
Hiện paper để trống `\ph{X}/80` và thừa nhận đây chỉ là prompt zero-shot. **Cần số thật.**

**Thiết kế:** inject fact sai **đã biết** vào sheet ở rate p có kiểm soát, theo từng loại
corruption, rồi đo **override rate** (narrative giữ fact sai = propagation, vs sửa về đúng-theo-video).

- Corruption có sẵn: `corrupt_scene` (đổi scene attr), `--p-obs-flip` (lật observation),
  `--p-scene`, `--p-obs-drop` trong `build_render_sft`. Cần đưa vào **đường inference** (hiện chỉ
  ở đường build training data) — 1 flag ở `text_dossier`.
- Đo propagation: dùng `claim_verify` probe fact bị đổi trong narrative output.

| Loại corruption | Rate inject | Override rate (sửa đúng) | Propagation (giữ sai) |
|---|---|---|---|
| Agent color flip | p | ? | X/80 |
| Root cause swap | p | ? | ? |
| Scene attr swap | p | ? | ? |

**Giá trị:** điền số `X/80`, định lượng rủi ro của consumption zero-shot, biến claim ở Fig.2 từ
giai thoại thành số đo. Đây là "honest negative characterization" reviewer đánh giá cao.

**Cần code:** flag inject corruption ở `text_dossier` (tái dùng `corrupt_scene`/flip có sẵn) +
checker propagation (`claim_verify`). Ước lượng: 1 ngày.

**Viết vào paper:** điền `X/80` ở §6 "Residual failures" + 1 câu định lượng ở caption Fig.2.

---

## E5 — Giá trị MBR/Verification + trần của pool  — Trung bình

**Support Stage 3 (+0.0092 nhỏ) và các negative results.**

### E5a — Oracle selection (trần của pool)
Trong `P_i`, chọn candidate có BERTScore-F1 **thật** cao nhất (so GT) → trần trên. So
`greedy` vs `MBR-selected` vs `oracle` ⇒ tách "chất lượng pool" khỏi "chất lượng selection",
cho biết còn bao nhiêu headroom (giải thích vì sao Stage 3 chỉ +0.0092).

| Chọn | QA | CL | SD | TD | Mean |
|---|---|---|---|---|---|
| greedy (Stage 2) | | | | | |
| MBR (Stage 3) | | | | | |
| oracle (trần) | | | | | |

**Cần code:** `mbr_select --oracle --gt val_gt.json` (chọn theo BERTScore thật). ~20 dòng.

### E5b — Tái lập negative result (ii): verify-on-letter
Apply verification fusion lên **closed MCQ letter** → đo net flip rate ~0. Xác nhận
"same-model verifier không flip được cái same-model misperceive". Củng cố tính trung thực của §5.3.

### E5c — MBR-vs-greedy overturn rate
Tần suất MBR lật greedy + delta ⇒ support "MBR frequently overturns the greedy choice".

**Viết vào paper:** 1 bảng oracle (E5a) trong §5.2/§5.3; số flip-rate (E5b/c) trong phần
negative results (đã có chỗ, chỉ điền số val).

---

## 2. Bảng ưu tiên & lộ trình

| # | Thí nghiệm | Vá luận điểm | Cần code | Effort | Ưu tiên |
|---|---|---|---|---|---|
| E1 | Gap verifier vs generator + reliability | §5.2 (trống), Eq.crossq, gate γ | vừa | 0.5–1 ngày | **Bắt buộc** |
| E2 | Ablation cô lập sheet (B0–B3 + leave-one-out) | Đóng góp chính, generality | vừa | 1 ngày | **Bắt buộc** |
| E3 | Chiều luồng P→G vs G→G | Thesis "none the other way" | vừa | 1 ngày | Cao |
| E4 | Override rate theo corruption | `X/80`, Fig.2, §6 | nhỏ (tái dùng) | 1 ngày | Cao |
| E5 | Oracle pool + MBR/greedy + verify-on-letter | Stage 3, negative (ii) | nhỏ | 0.5 ngày | Trung bình |

**Lộ trình đề xuất:** E1 + E2 trước (bắt buộc, đủ để paper phòng thủ được luận điểm trung tâm)
→ E4 (rẻ, điền placeholder) → E3 → E5.

## 3. Thay đổi cấu trúc paper đề xuất

- Thêm **§5.2 "Why it works: a controlled analysis on a held-out split"** (E1 + E2 + E3),
  đặt TRƯỚC bảng cumulative. Mở đầu bằng câu định vị val ở §0.
- Table 2 (cumulative) giữ nguyên = kết quả test cuối.
- §6 "Residual failures": điền `X/80` từ E4.
- Fig.2 caption: thêm 1 câu định lượng override từ E4.
- Negative results: điền flip-rate val từ E5b.

## 4. Rủi ro & lưu ý

- **Val ≠ test style:** luôn báo cáo delta/hành vi, không so tuyệt đối với test (đã framing ở §0).
- **Kích thước ~73 video:** đủ cho per-type mean nhưng nên báo cáo cùng khoảng tin cậy
  (bootstrap trên video) cho các delta nhỏ (E5), tránh over-claim.
- **Video root:** val video dưới `TRAIN_VIDEOS_ROOT`, không phải test root — `eval.sh` đã xử lý,
  nhưng khi chạy `phase3_infer.sh` trên val nhớ set `TEST_VIDEOS_ROOT=$TRAIN_VIDEOS_ROOT`.
- **Determinism:** mọi stage deterministic trừ sampling ở Stage 3/pool — fix seed để tái lập.
- **Không leak:** B2 oracle và E5a oracle dùng GT → chỉ là *upper-bound/analysis*, KHÔNG phải một
  cấu hình submit; ghi rõ trong caption để tránh hiểu nhầm.
```
