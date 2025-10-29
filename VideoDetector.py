# -*- coding: utf-8 -*-
"""
OpenCV CPU decoding, single open per video, buffered -> flush embedding/fingerprint.
- mode="clip": OpenCLIP ViT-L/14 embeddings (fp16 in RAM)
- mode="optical": OpticalImageCompare fingerprints
- 5 sample points, ±2 frames per point, 25-combo per point, all 5 must pass
- FRAME_BUFFER=1024 frames, BATCH_SIZE=32
- Prints device for CLIP
"""

import os, csv, base64
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8' # Suppresses most FFmpeg logs
import numpy as np
import cv2
from PIL import Image
from itertools import combinations
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

# ---------- config ----------
mode = "clip"                  # "clip" or "optical"
VIDEO_EXTS = (".mp4", ".avi")
SAMPLE_POINTS = 5
WINDOW_RADIUS = 2              # ±2 => 5 frames/point
SIM_THRESHOLD = 0.99 if mode == "clip" else 0.6
DURATION_TOL = 2.0             # seconds
FRAME_BUFFER = 1024            # fixed
BATCH_SIZE = 32                # fixed
EMB_DIM = 768                  # ViT-L/14

# ---------- ui ----------
Tk().withdraw()
folders = []
while True:
    d = filedialog.askdirectory(title="选择视频文件夹")
    if not d: break
    folders.append(d)
    if not messagebox.askyesno("继续？", "是否继续选择文件夹？"): break
if not folders: raise SystemExit("未选择任何文件夹")

# ---------- collect videos ----------
videos = []
for root in folders:
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(r, f))
if not videos: raise SystemExit("未找到视频文件")
print(f"视频数: {len(videos)}")

# ---------- helpers (opencv cpu) ----------
def video_meta_cv(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return 0.0, 0, 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-6 else 30.0
    dur = (n / fps) if n > 0 else 0.0
    cap.release()
    return dur, n, fps

def sample_indices(n, k=SAMPLE_POINTS):
    step = n // (k + 1)
    return [step * (i + 1) for i in range(k)]

def read_frames_single_open(path, indices_sorted_unique):
    """Open once, seek-and-read multiple frames. Return list aligned to indices_sorted_unique."""
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [None] * len(indices_sorted_unique)
    last_pos = -2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in indices_sorted_unique:
        idx = int(np.clip(idx, 0, max(total - 1, 0)))
        if idx != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last_pos = idx
    cap.release()
    return frames

# ---------- storage ----------
durations = {}
if mode == "clip":
    emb_store = {v: np.zeros((SAMPLE_POINTS, 1 + 2 * WINDOW_RADIUS, EMB_DIM), dtype=np.float16) for v in videos}
else:  # store fingerprint objects
    emb_store = {v: [[None]*(1 + 2*WINDOW_RADIUS) for _ in range(SAMPLE_POINTS)] for v in videos}
pending = []   # list of (path, sp, wi, ndarray_RGB)

# ---------- lazy model / optical ----------
_model = _preprocess = None
_device = None

def ensure_clip():
    global _model, _preprocess, _device
    if mode != "clip": return
    if _model is None:
        import torch, open_clip
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"当前使用设备: {_device}")
        if _device == "cuda":
            print(f"显卡名称: {torch.cuda.get_device_name(0)}")
        _model, _, _preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        _model.to(_device).eval()

ensure_clip()

if mode == "optical":
    import OpticalImageCompare as OIC

# ---------- flush ----------
def flush():
    if not pending: return
    if mode == "clip":
        import torch
        ensure_clip()
        tensors, metas = [], []
        for (path, sp, wi, arr) in pending:
            tensors.append(_preprocess(Image.fromarray(arr)))
            metas.append((path, sp, wi))
        pending.clear()
        with torch.no_grad():
            outs = []
            for s in range(0, len(tensors), BATCH_SIZE):
                batch = torch.stack(tensors[s:s+BATCH_SIZE]).to(_device, non_blocking=True)
                with torch.amp.autocast("cuda") if _device == "cuda" else torch.autocast(device_type="cpu"):
                    emb = _model.encode_image(batch)
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                outs.append(emb.detach().float().cpu().numpy().astype(np.float16))
            E = np.concatenate(outs, axis=0)
        for (path, sp, wi), vec in zip(metas, E):
            if path in emb_store:
                emb_store[path][sp, wi, :] = vec
    else:
        metas, fps = [], []
        for (path, sp, wi, arr) in pending:
            fp = OIC.image_to_fingerprint(Image.fromarray(arr))
            metas.append((path, sp, wi)); fps.append(fp)
        pending.clear()
        for (path, sp, wi), fp in zip(metas, fps):
            emb_store[path][sp][wi] = fp

# ---------- stage A: decode-only, single-open per video ----------
print(f"解码阶段（OpenCV CPU，单次打开/多帧读取，模式: {mode}）...")
for path in tqdm(videos):
    dur, nframes, fps = video_meta_cv(path)
    durations[path] = dur
    if nframes <= 0 or fps <= 0: continue

    # build target indices and metadata order
    pts = sample_indices(nframes, SAMPLE_POINTS)
    meta_pairs = []         # [(sp, wi), ...] aligned with order_list
    order_list = []         # [frame_index, ...]
    for i, pidx in enumerate(pts):
        for off in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            idx = int(np.clip(pidx + off, 0, nframes - 1))
            order_list.append(idx)
            meta_pairs.append((i, off + WINDOW_RADIUS))

    # unique sorted read, then map back
    uniq_sorted = sorted(set(order_list))
    idx2frame = {}
    frames = read_frames_single_open(path, uniq_sorted)
    for uidx, frm in zip(uniq_sorted, frames):
        idx2frame[uidx] = frm

    # restore original order and push to pending
    for (idx, (sp, wi)) in zip(order_list, meta_pairs):
        arr = idx2frame.get(idx, None)
        if arr is not None:
            pending.append((path, sp, wi, arr))

    if len(pending) >= FRAME_BUFFER:
        flush()

flush()  # final

# ---------- compare ----------
def cos_max25(a5, b5):
    A = a5.astype(np.float32); B = b5.astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A @ B.T).max()

def videos_equal_clip(a, b):
    if abs(durations.get(a, 0.0) - durations.get(b, 0.0)) > DURATION_TOL:
        return False
    Ea, Eb = emb_store[a], emb_store[b]   # (5,5,768)
    if not np.isfinite(Ea).all() or not np.isfinite(Eb).all(): return False
    for i in range(SAMPLE_POINTS):
        if cos_max25(Ea[i], Eb[i]) < SIM_THRESHOLD:
            return False
    return True

def videos_equal_optical(a, b):
    if abs(durations.get(a, 0.0) - durations.get(b, 0.0)) > DURATION_TOL:
        return False
    Ea, Eb = emb_store[a], emb_store[b]   # lists of fingerprints
    for i in range(SAMPLE_POINTS):
        matched = False
        for fa in Ea[i]:
            if fa is None: continue
            for fb in Eb[i]:
                if fb is None: continue
                if OIC.compare_fingerprints(fa, fb) >= SIM_THRESHOLD:
                    matched = True; break
            if matched: break
        if not matched:
            return False
    return True

cmp_func = videos_equal_clip if mode == "clip" else videos_equal_optical

print("比较阶段 ...")
groups = []
for i, j in tqdm(combinations(range(len(videos)), 2), total=len(videos)*(len(videos)-1)//2):
    a, b = videos[i], videos[j]
    try:
        if cmp_func(a, b):
            s = {a, b}
            merged = False
            for g in groups:
                if g & s:
                    g |= s; merged = True; break
            if not merged:
                groups.append(s)
    except Exception as e:
        print(f"比较失败 {a} vs {b}: {e}")

# ---------- save csv ----------
out_csv = f"duplicate_videos_{mode}_opencv_singleopen.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for g in groups:
        row = [base64.b64encode(p.encode()).decode() for p in sorted(g)]
        w.writerow(row)
print(f"完成 {len(groups)} 组，已保存 {out_csv}")
