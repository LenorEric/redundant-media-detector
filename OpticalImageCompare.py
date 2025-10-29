# requirements:
# pip install pillow numpy opencv-python scikit-image

import numpy as np
from PIL import Image
import cv2
import sys

try:
    from skimage.feature import local_binary_pattern
except Exception as e:
    print(f"[Error] 无法导入 scikit-image: {e}", file=sys.stderr)
    raise


# =========================================================
# 结构：DCT(256) + LBP(256) = 512维 → float16 → ≈1024 Byte
# =========================================================

def _to_gray_np(img_pil: Image.Image, size: int = 256) -> np.ndarray:
    """转灰度 + 统一尺寸"""
    try:
        g = img_pil.convert("L").resize((size, size), Image.BICUBIC)
        return np.asarray(g, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"图像灰度化或缩放失败: {e}") from e


def _dct256(gray256: np.ndarray) -> np.ndarray:
    """提取 DCT 低频特征 16×16=256 维"""
    try:
        f = cv2.dct(gray256.astype(np.float32) / 255.0)
        low = f[:16, :16].copy()
        low[0, 0] = 0.0
        vec = low.flatten()
        return vec.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"DCT 计算失败: {e}") from e


def _lbp256(gray256: np.ndarray) -> np.ndarray:
    """计算 LBP 直方图 256 维"""
    try:
        lbp = local_binary_pattern(gray256, P=8, R=1, method='default')
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        return hist.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"LBP 计算失败: {e}") from e


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
        f_dct = _dct256(gray)      # 256维
        f_lbp = _lbp256(gray)      # 256维
        feat = np.concatenate([f_dct, f_lbp], axis=0).astype(np.float32)
        feat = _l2_normalize(feat).astype(np.float16)
        return feat
    except Exception as e:
        raise RuntimeError(f"生成指纹向量失败: {e}") from e


def compare_fingerprints(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算相似度 ∈ [0,1]。
    采用余弦相似度并线性映射。
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
        imgA = Image.open("a.jpg")
        imgB = Image.open("c.jpg")
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
