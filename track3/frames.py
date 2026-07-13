"""Uniform / timestamped video frame extraction.

Used in ``extract`` frame-mode (see method.md §3.3). Pre-extracting frames makes
training I/O deterministic and fast across epochs, and lets us tell the model the
timestamp of each frame for the temporal-localization task.

Decoding backend preference: PyAV/torchvision/decord are heavy; we use OpenCV,
which is already a project dependency and handles the 8 source containers well.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2


@dataclass
class FrameResult:
    paths: list[str]        # extracted JPEG paths, in temporal order
    timestamps: list[float]  # seconds for each frame
    duration: float          # video duration in seconds


def _video_meta(cap: "cv2.VideoCapture") -> tuple[int, float]:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return n, fps


def extract_uniform_frames(
    video_path: str,
    out_dir: str,
    num_frames: int = 16,
    max_side: int = 448,
    overwrite: bool = False,
) -> FrameResult:
    """Sample ``num_frames`` evenly across the video and write JPEGs.

    Returns frame paths, per-frame timestamps (s) and duration (s). Idempotent:
    re-uses existing frames unless ``overwrite`` is set.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n_total, fps = _video_meta(cap)
    duration = (n_total / fps) if fps > 0 else 0.0

    if n_total <= 0:  # some containers don't report frame count; fall back to a stream read
        return _extract_streaming(cap, out_dir, num_frames, max_side, fps, overwrite)

    indices = _even_indices(n_total, num_frames)
    paths, stamps = [], []
    for i, idx in enumerate(indices):
        dst = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        ts = (idx / fps) if fps > 0 else 0.0
        if not overwrite and os.path.exists(dst):
            paths.append(dst); stamps.append(ts); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(dst, _resize(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(dst); stamps.append(ts)
    cap.release()
    return FrameResult(paths, stamps, duration)


def _extract_streaming(cap, out_dir, num_frames, max_side, fps, overwrite) -> FrameResult:
    """Fallback when frame count is unknown: read sequentially, keep every k-th."""
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded")
    indices = _even_indices(len(frames), num_frames)
    duration = (len(frames) / fps) if fps > 0 else 0.0
    paths, stamps = [], []
    for i, idx in enumerate(indices):
        dst = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        if overwrite or not os.path.exists(dst):
            cv2.imwrite(dst, _resize(frames[idx], max_side), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(dst)
        stamps.append((idx / fps) if fps > 0 else 0.0)
    return FrameResult(paths, stamps, duration)


def _even_indices(total: int, k: int) -> list[int]:
    if total <= k:
        return list(range(total))
    step = total / k
    return [min(total - 1, int(i * step + step / 2)) for i in range(k)]


def _resize(frame, max_side: int):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def format_timestamp(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def safe_name(video_id: str) -> str:
    """Filesystem-safe slug for a (possibly nested) video_id."""
    return video_id.replace("/", "__").replace(".", "_")


def video_duration(video_path: str) -> float:
    """Best-effort video duration in seconds (0.0 if it can't be probed)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    n, fps = _video_meta(cap)
    cap.release()
    return (n / fps) if fps > 0 else 0.0


def duration_hint(duration: float) -> str:
    """Total-duration hint for native-``video``-mode temporal items.

    In native video mode Qwen3-VL keeps the real fps/duration and time-aligned
    position IDs (unlike an image list — see :func:`track3.tasks.frame_plan`), so
    the prompt only needs the wall-clock length to let the model emit MM:SS. Shared
    by build_dataset (train) and infer (test) so the temporal framing never drifts.
    """
    if duration <= 0:
        return ""
    return (
        f"Video duration: {format_timestamp(duration)} (MM:SS). "
        f"Identify when the queried event starts and ends; both timestamps must "
        f"lie within 00:00–{format_timestamp(duration)} and start ≤ end."
    )


def timestamp_hint(stamps: list[float], duration: float) -> str:
    """Explicit time-axis hint injected into the prompt for temporal tasks.

    Gives the model the discrete timestamps of the frames it is shown so it can
    anchor the queried event to wall-clock ``MM:SS`` (the temporal-localization
    lever, see method.md §3.3). The instruction nudges the model to pick start/end
    *from / between* the visible frame timestamps rather than guessing — without
    changing the ``{"start","end"}`` output schema the IoU grader expects.
    """
    sampled = ", ".join(format_timestamp(s) for s in stamps)
    return (
        f"Video duration: {format_timestamp(duration)} (MM:SS). "
        f"The {len(stamps)} frames above are sampled, in order, at these "
        f"timestamps: {sampled}. "
        f"Decide which timestamp the queried event starts at and which it ends at; "
        f"both must lie within 00:00–{format_timestamp(duration)} and start ≤ end."
    )
