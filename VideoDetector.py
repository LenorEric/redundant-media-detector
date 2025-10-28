import os, csv, base64
import numpy as np
import torch
import av
from PIL import Image
from itertools import combinations
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm
import open_clip
import OpticalImageCompare as OIC   # 新增模块

# ---------------- config ----------------
mode = "clip"       # "clip" 或 "optical"
VIDEO_EXTS = (".mp4", ".avi")
SAMPLE_POINTS = 5
WINDOW_RADIUS = 2
SIM_THRESHOLD = 0.99
DURATION_TOL = 2.0
FRAME_BUFFER = 1024
BATCH_SIZE = 32
EMB_DIM = 768

# ---------------- ui ----------------
Tk().withdraw()
folders = []
while True:
    d = filedialog.askdirectory(title="选择视频文件夹")
    if not d: break
    folders.append(d)
    if not messagebox.askyesno("继续？", "是否继续选择文件夹？"): break
if not folders: raise SystemExit("未选择文件夹")

# ---------------- collect videos ----------------
videos = []
for root in folders:
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(r, f))
if not videos: raise SystemExit("未找到视频文件")
print(f"视频数: {len(videos)}")

# ---------------- helpers ----------------
def video_meta(path):
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else (float(s.r_frame_rate) if s.r_frame_rate else 30.0)
            if s.frames and s.frames > 0:
                n = int(s.frames)
                dur = n / fps
            else:
                dur = (c.duration / 1e6) if c.duration else 0.0
                n = int(round(dur * fps))
            return dur, n, fps
    except Exception:
        return 0.0, 0, 0.0

def sample_indices(n, k=SAMPLE_POINTS):
    step = n // (k + 1)
    return [step * (i + 1) for i in range(k)]

def grab_frame(container, stream, idx, fps):
    ts_us = int((idx / fps) * 1e6)
    container.seek(ts_us, any_frame=False, backward=True, stream=stream)
    for frame in container.decode(stream):
        return frame.to_rgb().to_ndarray()
    return None

# ---------------- storage ----------------
durations = {}
pending = []   # (path, sp, wi, ndarray)
emb_store = {v: np.zeros((SAMPLE_POINTS, 1 + 2 * WINDOW_RADIUS, EMB_DIM), dtype=np.float16) for v in videos}

# ---------------- model (lazy init) ----------------
_model = None
_preprocess = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_model():
    global _model, _preprocess
    if mode != "clip": return
    if _model is None:
        print(f"当前使用设备: {_device}")
        if _device == "cuda":
            print(f"显卡名称: {torch.cuda.get_device_name(0)}")
        _model, _, _preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        _model.to(_device).eval()

# ---------------- flush ----------------
def flush():
    if not pending: return
    if mode == "clip":
        ensure_model()
        tensors, metas = [], []
        for (path, sp, wi, arr) in pending:
            img = Image.fromarray(arr)
            tensors.append(_preprocess(img))
            metas.append((path, sp, wi))
        pending.clear()
        with torch.no_grad():
            out_chunks = []
            for s in range(0, len(tensors), BATCH_SIZE):
                batch = torch.stack(tensors[s:s+BATCH_SIZE]).to(_device, non_blocking=True)
                with torch.amp.autocast("cuda") if _device == "cuda" else torch.autocast(device_type="cpu"):
                    emb = _model.encode_image(batch)
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                out_chunks.append(emb.detach().float().cpu().numpy().astype(np.float16))
            E = np.concatenate(out_chunks, axis=0)
        for (path, sp, wi), vec in zip(metas, E):
            if path in emb_store:
                emb_store[path][sp, wi, :] = vec

    elif mode == "optical":
        metas, fps = [], []
        for (path, sp, wi, arr) in pending:
            img = Image.fromarray(arr)
            fp = OIC.image_to_fingerprint(img)
            metas.append((path, sp, wi))
            fps.append(fp)
        pending.clear()
        # fingerprint 可以直接存成 object 列表，这里假设长度不定，使用object类型容器
        for (path, sp, wi), fp in zip(metas, fps):
            emb_store[path][sp, wi, 0:len(fp)] = np.frombuffer(fp, dtype=np.float16)[:EMB_DIM]

# ---------------- decode ----------------
print(f"解码阶段（模式: {mode}）...")
for path in tqdm(videos):
    dur, nframes, fps = video_meta(path)
    durations[path] = dur
    if nframes <= 0 or fps <= 0: continue
    pts = sample_indices(nframes, SAMPLE_POINTS)
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        for i, pidx in enumerate(pts):
            for off in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
                idx = int(np.clip(pidx + off, 0, nframes - 1))
                arr = grab_frame(container, stream, idx, fps)
                if arr is not None:
                    pending.append((path, i, off + WINDOW_RADIUS, arr))
        container.close()
    except Exception as e:
        print(f"读取失败 {path}: {e}")

    if len(pending) >= FRAME_BUFFER:
        flush()

flush()

# ---------------- compare ----------------
def cos_max25(a5, b5):
    A = a5.astype(np.float32); B = b5.astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A @ B.T).max()

def videos_equal(a, b):
    if abs(durations[a] - durations[b]) > DURATION_TOL:
        return False
    Ea, Eb = emb_store[a], emb_store[b]
    if mode == "clip":
        for i in range(SAMPLE_POINTS):
            if cos_max25(Ea[i], Eb[i]) < SIM_THRESHOLD:
                return False
        return True
    elif mode == "optical":
        for i in range(SAMPLE_POINTS):
            matched = False
            for pa in Ea[i]:
                for pb in Eb[i]:
                    sim = OIC.compare_fingerprints(pa, pb)
                    if sim >= SIM_THRESHOLD:
                        matched = True
                        break
                if matched: break
            if not matched:
                return False
        return True

# ---------------- match & save ----------------
print("比较阶段 ...")
groups = []
for i, j in tqdm(combinations(range(len(videos)), 2), total=len(videos)*(len(videos)-1)//2):
    a, b = videos[i], videos[j]
    try:
        if videos_equal(a, b):
            s = {a, b}
            merged = False
            for g in groups:
                if g & s:
                    g |= s; merged = True; break
            if not merged:
                groups.append(s)
    except Exception as e:
        print(f"比较失败 {a} vs {b}: {e}")

out_csv = f"duplicate_videos_{mode}.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for g in groups:
        row = [base64.b64encode(p.encode()).decode() for p in sorted(g)]
        w.writerow(row)
print(f"完成 {len(groups)} 组，结果保存为 {out_csv}")
