import os, csv, base64
import cv2, numpy as np
from PIL import Image
from tkinter import Tk, filedialog, messagebox, simpledialog
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

import torch, open_clip

# ---------------- Params ----------------
SAMPLE_POINTS   = 5
WINDOW_RADIUS   = 2              # ±2 -> 5帧
SIM_THRESHOLD   = 0.99
DURATION_TOL    = 2.0            # seconds
VIDEO_EXTS      = (".mp4", ".avi")
FRAME_BUFFER    = 1024           # 到此数量就触发一次编码
EMB_BATCH       = 64             # 模型微批
MODEL_NAME      = "ViT-L-14"
EMB_DIM         = 768            # ViT-L/14
TARGET_SIZE     = (224, 224)     # 与 open_clip 预处理一致

# ---------------- UI ----------------
Tk().withdraw()
folders = []
while True:
    d = filedialog.askdirectory(title="选择视频文件夹")
    if not d: break
    folders.append(d)
    if not messagebox.askyesno("继续？", "还要选择其他文件夹吗？"): break
if not folders: raise SystemExit("未选择文件夹")

buf_input = simpledialog.askinteger("帧缓冲", "累计多少帧再编码？(建议 256~4096)", minvalue=1, maxvalue=100000) or FRAME_BUFFER
buf_batch = simpledialog.askinteger("嵌入微批", "模型微批大小？(建议 16~128)", minvalue=1, maxvalue=4096) or EMB_BATCH
print(f"FRAME_BUFFER={buf_input}, EMB_BATCH={buf_batch}")

# ---------------- Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
model.to(device).eval()

# ---------------- Collect videos ----------------
videos = []
for root in folders:
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(r, f))
if not videos: raise SystemExit("未发现视频")
print(f"共 {len(videos)} 个视频")

# ---------------- Utils ----------------
def meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return 0,0,0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if fps > 1e-6 else 30.0
    dur = n / fps if n>0 else 0
    cap.release()
    return dur, n, fps

def sample_indices(n, k=SAMPLE_POINTS):
    step = n // (k + 1)
    return [step*(i+1) for i in range(k)]

# 统一的归一化，等价 open_clip 的 ToTensor+Normalize
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def to_clip_tensor_uint8(img_uint8):
    # img_uint8: HxWx3 uint8 RGB(0..255)
    t = torch.from_numpy(img_uint8).permute(2,0,1).contiguous()  # 3xHxW, uint8
    return t  # 延迟到批处理时再 /255、标准化、转半精度/上GPU

def normalize_batch_uint8_to_fp16(batch_u8):
    # batch_u8: Bx3xHxW uint8
    x = batch_u8.float().div_(255.0)
    # 标准化
    mean = torch.tensor(MEAN, device=x.device)[:, None, None]
    std  = torch.tensor(STD,  device=x.device)[:, None, None]
    x = (x - mean) / std
    return x.half()  # fp16 进入模型

def cos_max25(a5, b5):
    A = a5.astype(np.float32); B = b5.astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True)+1e-8)
    B /= (np.linalg.norm(B, axis=1, keepdims=True)+1e-8)
    return (A @ B.T).max()

# ---------------- Stage A: plan frames with bounded buffer ----------------
# RAM 存：每视频一个容器 (5,5,EMB_DIM) fp16；先放占位，全0
emb_store = {p: np.zeros((SAMPLE_POINTS, 1+2*WINDOW_RADIUS, EMB_DIM), dtype=np.float16) for p in videos}
durations = {}
# 暂存“待编码的帧”：uint8(224x224x3) + 元信息
pending_imgs = []      # list of np.uint8 HxWx3
pending_meta = []      # list of (path, sp, wi)

def flush_encode():
    if not pending_imgs: return
    # ---- 预处理到张量，并分微批送模型 ----
    # 拼成一个大 batch 的 uint8
    u8 = torch.stack([to_clip_tensor_uint8(img) for img in pending_imgs])  # Bx3xHxW uint8
    # 分块送入 GPU
    with torch.no_grad():
        out_chunks = []
        for s in range(0, u8.size(0), buf_batch):
            u = u8[s:s+buf_batch].to(device, non_blocking=True)
            x = normalize_batch_uint8_to_fp16(u)  # Bx3xHxW fp16
            with torch.amp.autocast("cuda"):
                e = model.encode_image(x)
            e = torch.nn.functional.normalize(e, dim=-1)
            out_chunks.append(e.detach().float().cpu().numpy().astype(np.float16))
        E = np.concatenate(out_chunks, axis=0)  # BxD fp16

    # 回填
    for (path, sp, wi), vec in zip(pending_meta, E):
        if path in emb_store:
            emb_store[path][sp, wi, :] = vec

    pending_imgs.clear()
    pending_meta.clear()

print("Planning frames and encoding in chunks...")
for path in tqdm(videos):
    dur, nframes, fps = meta(path)
    durations[path] = dur
    if nframes <= 0:
        continue
    pts = sample_indices(nframes, SAMPLE_POINTS)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): continue

    for i, p in enumerate(pts):
        for off in range(-WINDOW_RADIUS, WINDOW_RADIUS+1):
            idx = int(np.clip(p+off, 0, nframes-1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 早缩放，显著降内存
            rgb = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            pending_imgs.append(rgb)                  # uint8, 小
            pending_meta.append((path, i, off+WINDOW_RADIUS))

            if len(pending_imgs) >= buf_input:
                flush_encode()  # 达阈值就编码一次

    cap.release()

# 收尾编码
flush_encode()

# ---------------- Stage B: compare after all embeddings ready ----------------
def videos_equal(pa, pb):
    if abs(durations.get(pa, 0) - durations.get(pb, 0)) > DURATION_TOL:
        return False
    Ea, Eb = emb_store[pa], emb_store[pb]
    if (Ea == 0).all() or (Eb == 0).all():   # 有失败留下的全0
        return False
    for i in range(SAMPLE_POINTS):
        if cos_max25(Ea[i], Eb[i]) < SIM_THRESHOLD:
            return False
    return True

print("Comparing...")
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
        print(f"Compare failed: {a} vs {b}: {e}")

# ---------------- CSV ----------------
out_csv = "duplicate_videos.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for g in groups:
        row = [base64.b64encode(p.encode()).decode() for p in sorted(g)]
        w.writerow(row)
print(f"完成 {len(groups)} 组，已保存 {out_csv}")
