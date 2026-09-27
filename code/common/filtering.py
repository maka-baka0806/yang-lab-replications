"""
体素级放射组学滤波（Radiomic Filtering）
=========================================
对应文献：
  - Yang Z, et al. Quantification of lung function on CT images based on
    pulmonary radiomic filtering. Med Phys 2022;49(11):7278-7286.
  - Zhang R, ..., Yang Z. An Explainable Neural Radiomic Sequence Model with
    Spatiotemporal Continuity for Quantifying 4DCT-based Pulmonary Ventilation.
    arXiv:2503.23898 (2025).

核心思想：传统放射组学把整个器官当成一个 ROI，得到一个特征向量；
本方法把 3D 滑窗在器官内逐体素滑动，每个位置算一组特征，
于是每个特征变成一张「特征图」——具有空间分辨能力。

实现要点（与原论文一致的细节）：
  - 滑窗核大小可调（原论文 15×15×15 mm³ / 26×26×26 mm³）
  - 13 个方向取平均以近似旋转不变性
  - 全部计算向量化（scipy.ndimage 可分离滤波），因此比逐窗循环快数百倍
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# 13 个方向：3 个轴向 + 6 个面对角线 + 4 个体对角线（与原论文取平均的方向集一致）
DIRECTIONS_13 = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1),
    (1, 0, -1), (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]

FEATURE_LABELS = {
    "mean": "一阶 · 局部均值",
    "std": "一阶 · 局部标准差",
    "entropy": "一阶 · 局部熵",
    "uniformity": "一阶 · 局部均匀度",
    "contrast": "纹理 · 局部对比度（13 方向近似 GLCM Contrast）",
    "homogeneity": "纹理 · 局部同质性（13 方向近似 GLCM Homogeneity）",
    "gradient": "梯度 · 边缘强度",
}


@dataclass
class FilterResult:
    maps: dict[str, np.ndarray]
    mask: np.ndarray
    kernel_size: int
    bin_count: int = 32
    meta: dict = field(default_factory=dict)


def _binned(volume: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    """在 ROI 内做灰度离散化（对应 IBSI 的 fixed bin number 策略）。"""
    vals = volume[mask]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    idx = np.clip(((volume - lo) / (hi - lo) * (bins - 1)).round(), 0, bins - 1)
    return idx.astype(np.int16) + 1     # 0 留给背景


def _shift(volume: np.ndarray, d: tuple[int, int, int]) -> np.ndarray:
    """按方向 d 平移体数据（用于 GLCM 类纹理的向量化计算）。"""
    out = np.zeros_like(volume)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for ax, step in enumerate(d):
        if step == 1:
            src[ax] = slice(0, -1)
            dst[ax] = slice(1, None)
        elif step == -1:
            src[ax] = slice(1, None)
            dst[ax] = slice(0, -1)
    out[tuple(dst)] = volume[tuple(src)]
    return out


def radiomic_filtering(
    volume: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 15,
    bins: int = 32,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    features: list[str] | None = None,
) -> FilterResult:
    """对 volume 在 mask 内做体素级放射组学滤波，返回各特征的空间图。"""
    features = features or ["mean", "std", "entropy", "uniformity",
                            "contrast", "homogeneity", "gradient"]
    vol = volume.astype(np.float32)

    # 核大小按体素计（考虑 spacing）
    k = max(3, int(round(kernel_size / max(spacing[0], 1e-6))))
    if k % 2 == 0:
        k += 1
    size = (k, k, k)

    out: dict[str, np.ndarray] = {}

    # ---- 一阶特征 ----
    mean = ndimage.uniform_filter(vol, size=size, mode="nearest")
    mean_sq = ndimage.uniform_filter(vol ** 2, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean ** 2, 0)

    if "mean" in features:
        out["mean"] = mean
    if "std" in features:
        out["std"] = np.sqrt(var)

    # 局部熵 / 均匀度：对每个灰度级做指示函数滤波（可分离，速度快）
    if "entropy" in features or "uniformity" in features:
        idx = _binned(vol, mask, bins)
        p_sum = np.zeros_like(vol)
        p_sq_sum = np.zeros_like(vol)
        for b in range(1, bins + 1):
            ind = (idx == b).astype(np.float32)
            p = ndimage.uniform_filter(ind, size=size, mode="nearest")
            p_sum += np.where(p > 0, -p * np.log(p + 1e-12), 0.0)
            p_sq_sum += p ** 2
        if "entropy" in features:
            out["entropy"] = p_sum
        if "uniformity" in features:
            out["uniformity"] = p_sq_sum

    # ---- 纹理特征：13 方向平均 ----
    if "contrast" in features or "homogeneity" in features:
        c_acc = np.zeros_like(vol)
        h_acc = np.zeros_like(vol)
        for d in DIRECTIONS_13:
            diff = vol - _shift(vol, d)
            d2 = diff ** 2
            c_acc += ndimage.uniform_filter(d2, size=size, mode="nearest")
            h_acc += ndimage.uniform_filter(1.0 / (1.0 + d2), size=size, mode="nearest")
        if "contrast" in features:
            out["contrast"] = c_acc / len(DIRECTIONS_13)
        if "homogeneity" in features:
            out["homogeneity"] = h_acc / len(DIRECTIONS_13)

    # ---- 梯度 ----
    if "gradient" in features:
        g = np.gradient(vol)
        out["gradient"] = np.sqrt(sum(gi ** 2 for gi in g))

    # ROI 外清零
    for key in out:
        out[key] = np.where(mask, out[key], np.nan)

    return FilterResult(maps=out, mask=mask, kernel_size=k, bin_count=bins,
                        meta={"voxel_kernel": k})


# ----------------------------------------------------------------------
# 合成肺体模：用于复现「特征图 vs 功能成像」的相关性实验
# ----------------------------------------------------------------------
@dataclass
class LungPhantom:
    ct: np.ndarray            # 模拟 CT（HU）
    mask: np.ndarray          # 肺掩膜
    reference: np.ndarray     # 参考通气图（相当于 Galligas PET / DTPA SPECT）
    defect: np.ndarray        # 通气缺损区
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


def make_lung_phantom(size: int = 64, defect_side: str = "left",
                      defect_radius: int = 9, noise: float = 12.0,
                      seed: int = 0) -> LungPhantom:
    """构造一个带通气缺损的合成肺。

    - 健康肺区：低密度（约 -850 HU）、纹理细密均匀
    - 通气缺损区：密度升高（约 -720 HU）、纹理粗糙（模拟气体潴留）
    - reference：连续的通气强度图，缺损区显著降低
    """
    rng = np.random.RandomState(seed)
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]

    # 两叶肺：两个椭球
    left = (((xx - c + 13) / 11) ** 2 + ((yy - c) / 13) ** 2 + ((zz - c) / 15) ** 2) <= 1
    right = (((xx - c - 13) / 11) ** 2 + ((yy - c) / 13) ** 2 + ((zz - c) / 15) ** 2) <= 1
    mask = left | right

    # 缺损区
    if defect_side == "left":
        dc = (c - 13, c, c)
    elif defect_side == "right":
        dc = (c + 13, c, c)
    else:
        dc = (c, c, c)
    defect = (((xx - dc[0]) ** 2 + (yy - dc[1]) ** 2 + (zz - dc[2]) ** 2)
              <= defect_radius ** 2) & mask

    # ---- CT 强度 ----
    ct = np.full((size, size, size), -1000.0, dtype=np.float32)
    base = -850.0 + rng.randn(size, size, size) * noise
    ct[mask] = base[mask]
    # 缺损区：密度升高 + 更强的纹理起伏
    rough = -720.0 + rng.randn(size, size, size) * (noise * 2.4)
    ct[defect] = rough[defect]
    # 加一点低频空间变化，使纹理图更有结构
    low = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 4) * 45
    ct[mask] += low[mask]

    # ---- 参考通气图 ----
    ref = np.zeros((size, size, size), dtype=np.float32)
    ref[mask] = 1.0
    ref[defect] = 0.18
    ref = ndimage.gaussian_filter(ref, 2.5)
    ref = np.where(mask, ref, np.nan)

    return LungPhantom(ct=ct, mask=mask, reference=ref, defect=defect)


# ----------------------------------------------------------------------
# 相关性分析（对应原论文的体素级 Spearman 相关）
# ----------------------------------------------------------------------
def spearman_map_vs_reference(feature_map: np.ndarray, reference: np.ndarray,
                              mask: np.ndarray) -> float:
    """特征图与参考功能图的体素级 Spearman 相关（在 ROI 内）。"""
    from scipy.stats import spearmanr

    a = feature_map[mask]
    b = reference[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return float("nan")
    rho, _ = spearmanr(a[ok], b[ok])
    return float(rho)


def rank_features(result: FilterResult, reference: np.ndarray) -> list[dict]:
    """对全部特征图计算与参考图的相关性并排序。"""
    rows = []
    for name, fmap in result.maps.items():
        rho = spearman_map_vs_reference(fmap, reference, result.mask)
        rows.append({
            "特征": FEATURE_LABELS.get(name, name),
            "key": name,
            "Spearman ρ": round(rho, 4) if np.isfinite(rho) else float("nan"),
            "abs_rho": abs(rho) if np.isfinite(rho) else -1.0,
        })
    return sorted(rows, key=lambda r: -r["abs_rho"])
