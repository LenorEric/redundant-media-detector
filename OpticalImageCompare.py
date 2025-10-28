# requirements:
# pip install pillow numpy opencv-python torch torchvision scikit-image

from typing import Tuple
import numpy as np
from PIL import Image
import cv2
import sys

# --- 可选：CNN 中层特征 ---
_CNN_OK = False
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    _CNN_WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
    _cnn_model = mobilenet_v3_small(weights=_CNN_WEIGHTS).features.eval()
    _CNN_OK = True
except Exception as e:
    _CNN_OK = False
    print(f"[Warning] Torch/TorchVision 初始化失败: {e}", file=sys.stderr)
    print("[Info] CNN 特征提取将被禁用（结果精度会下降）", file=sys.stderr)

# --- LBP（256维） ---
try:
    from skimage.feature import local_binary_pattern
except Exception as e:
    print(f"[Error] 无法导入 scikit-image: {e}", file=sys.stderr)
    raise

# =========================================================
# 目标：返回 1024 Byte 指纹向量（float16 × 512 维 = 1024B）
# 结构：DCT(128) + LBP(256) + CNN(128)
# =========================================================

def _to_gray_np(img_pil: Image.Image, size: int = 256) -> np.ndarray:
    try:
        g = img_pil.convert("L").resize((size, size), Image.BICUBIC)
        return np.asarray(g, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"图像灰度化或缩放失败: {e}") from e

def _to_rgb_tensor(img_pil: Image.Image, size: int = 224):
    if not _CNN_OK:
        return None
    try:
        tfm = _CNN_WEIGHTS.transforms()
        img = img_pil.convert("RGB").resize((size, size), Image.BICUBIC)
        return tfm(img).unsqueeze(0)  # [1,3,224,224]
    except Exception as e:
        print(f"[Warning] 图像转换为 Torch 张量失败: {e}", file=sys.stderr)
        return None

def _dct128(gray256: np.ndarray) -> np.ndarray:
    try:
        f = cv2.dct(gray256.astype(np.float32) / 255.0)
        low = f[:12, :12].copy()
        low[0, 0] = 0.0
        vec = low.flatten()[:128]
        return vec.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"DCT 计算失败: {e}") from e

def _lbp256(gray256: np.ndarray) -> np.ndarray:
    try:
        lbp = local_binary_pattern(gray256, P=8, R=1, method='default')
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        return hist.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"LBP 计算失败: {e}") from e

def _cnn128(img_pil: Image.Image) -> np.ndarray:
    if not _CNN_OK:
        return np.zeros(128, dtype=np.float32)
    try:
        with torch.no_grad():
            x = _to_rgb_tensor(img_pil)
            if x is None:
                raise RuntimeError("Torch 图像张量为空。")
            feats = _cnn_model(x)
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(0).squeeze(-1).squeeze(-1)
            vec = feats.cpu().numpy().astype(np.float32)
            if vec.shape[0] >= 128:
                vec = vec[:128]
            else:
                pad = np.zeros(128 - vec.shape[0], dtype=np.float32)
                vec = np.concatenate([vec, pad], axis=0)
            return vec
    except Exception as e:
        print(f"[Warning] CNN 特征提取失败: {e}", file=sys.stderr)
        return np.zeros(128, dtype=np.float32)

def _l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def image_to_fingerprint(img_pil: Image.Image) -> np.ndarray:
    """
    输入：PIL.Image
    输出：np.ndarray，shape=(512,), dtype=float16 —— 约 1024 Byte
    """
    try:
        gray = _to_gray_np(img_pil, size=256)
        f_dct = _dct128(gray)
        f_lbp = _lbp256(gray)
        f_cnn = _cnn128(img_pil)
        feat = np.concatenate([f_dct, f_lbp, f_cnn], axis=0).astype(np.float32)
        feat = _l2_normalize(feat).astype(np.float16)
        return feat
    except Exception as e:
        raise RuntimeError(f"生成指纹向量失败: {e}") from e

def compare_fingerprints(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    返回相似度 ∈ [0,1]。使用余弦相似度并线性映射。
    """
    try:
        v1 = v1.astype(np.float32)
        v2 = v2.astype(np.float32)
        v1 = _l2_normalize(v1)
        v2 = _l2_normalize(v2)
        cos = float(np.dot(v1, v2))
        cos = max(min(cos, 1.0), -1.0)
        return 0.5 * (cos + 1.0)
    except Exception as e:
        raise RuntimeError(f"指纹相似度计算失败: {e}") from e

# =========================
# 示例
# =========================
if __name__ == "__main__":
    from pathlib import Path
    try:
        imgA = Image.open("a.png")
        imgB = Image.open("b.png")
    except Exception as e:
        print(f"[Error] 图像加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fpA = image_to_fingerprint(imgA)
        fpB = image_to_fingerprint(imgB)
        sim = compare_fingerprints(fpA, fpB)
        print("similarity:", sim)
    except Exception as e:
        print(f"[Fatal] 计算失败: {e}", file=sys.stderr)
        sys.exit(2)
