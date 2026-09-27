"""
分割与不确定性量化（Segmentation & Uncertainty Quantification）
================================================================
对应文献：
  - Yang Z, et al. Quantifying U-Net uncertainty in multi-parametric MRI-based
    glioma segmentation by spherical image projection. Med Phys 2024;51(3):1931-1943.
    （SPU-Net：多投影 → 多组预测 → 方差即不确定性）
  - Yang Z, et al. A voxel-wise uncertainty-guided framework for glioma
    segmentation using spherical projection-based U-Net and localized refinement.
    Med Phys 2026;53(3):e70360.
  - Wang L, ..., Yang Z, et al. Uncertainty quantification in multi-parametric
    MRI-based meningioma radiotherapy target segmentation. Front Oncol 2025;15:1474590.
  - 国家自然科学基金青年项目：放疗中腹部图像自动分割的不确定性量化与评估研究

本模块用经典分割算法复现同一套「不确定性」范式，无需训练即可理解：
  1. 对同一影像做 N 次「视角扰动」（旋转/缩放/加噪）
  2. 每次得到一组分割结果
  3. 逐体素统计：均值 = 最终分割依据，方差/熵 = 不确定性
  4. 用 Otsu 法把不确定性图聚合成最终二值分割
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

METHODS = {
    "otsu": "Otsu 全局阈值（最常用基线）",
    "li": "Li 最小交叉熵阈值",
    "yen": "Yen 阈值",
    "adaptive": "自适应局部阈值（应对强度不均）",
    "kmeans": "K-means 聚类分割",
    "regiongrow": "区域生长（从种子点扩散）",
    "watershed": "分水岭（基于梯度）",
}


# ----------------------------------------------------------------------
# 合成病灶体模：带强度不均匀场的「类肿瘤」目标，用于对比分割算法
# ----------------------------------------------------------------------
@dataclass
class LesionPhantom:
    image: np.ndarray          # 模拟影像（HU）
    lesion: np.ndarray         # 真值：病灶
    organ: np.ndarray          # 真值：器官（背景组织）
    bias: np.ndarray           # 强度不均匀场（模拟线圈/射束硬化）
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


def make_lesion_phantom(size: int = 56, radius: int = 9, contrast: float = 60.0,
                        noise: float = 12.0, bias_strength: float = 0.35,
                        irregularity: float = 0.18, seed: int = 0) -> LesionPhantom:
    """构造带「强度不均匀 + 噪声 + 不规则边界」的病灶体模。

    这三样正是真实分割任务难的地方，也解释了为什么需要 U-Net 与不确定性量化：
      - bias_strength：低频强度漂移（阈值法会失效 → 自适应阈值占优）
      - noise：噪声（影响边界判定）
      - irregularity：边界不规则（形状先验失效）
    """
    rng = np.random.RandomState(seed)
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]

    # 器官：大球
    organ = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= (size * 0.42) ** 2

    # 病灶：球 + 球面随机扰动
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)
    noise_r = ndimage.gaussian_filter(rng.randn(size, size, size), 1.5) * radius * irregularity
    lesion = (r <= radius + noise_r) & organ

    # 强度不均匀场（低频）
    bias = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 9)
    bias = bias / (np.abs(bias).max() + 1e-6) * bias_strength

    image = np.zeros((size, size, size), dtype=np.float32)
    image[organ] = 40.0
    image[lesion] = 40.0 + contrast
    image[organ] *= (1.0 + bias[organ])
    image[organ] += rng.randn(int(organ.sum())) * noise

    return LesionPhantom(image=image, lesion=lesion, organ=organ,
                         bias=bias, spacing=(1.0, 1.0, 1.0))


# ----------------------------------------------------------------------
# 分割算法
# ----------------------------------------------------------------------
def _normalize(vol: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    v = vol.astype(np.float32)
    if roi is not None and roi.sum() > 0:
        lo, hi = np.percentile(v[roi], [0.5, 99.5])
    else:
        lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((v - lo) / (hi - lo), 0, 1)


def segment(volume: np.ndarray, method: str = "otsu",
            threshold: float | None = None, roi: np.ndarray | None = None,
            seed: tuple[int, int, int] | None = None,
            k: int = 2, tol: float = 0.20) -> np.ndarray:
    """对体数据做分割，返回布尔掩膜。

    roi：器官/感兴趣区掩膜。**真实流程都在 ROI 内分割**——
         没有它，阈值法会把背景（空气、体外）也算成前景，
         这是初学者最常见的错误，也是本函数强制建议传 roi 的原因。
    """
    from skimage import filters, segmentation as skseg

    norm = _normalize(volume, roi)
    roi = np.ones_like(norm, dtype=bool) if roi is None else roi
    vals = norm[roi]

    if method in ("otsu", "li", "yen"):
        if threshold is not None:
            t = threshold
        elif method == "otsu":
            t = filters.threshold_otsu(vals)
        elif method == "li":
            t = filters.threshold_li(vals)
        else:
            t = filters.threshold_yen(vals)
        return (norm > t) & roi

    if method == "adaptive":
        t = filters.threshold_local(norm, block_size=15, offset=0.0)
        return (norm > t) & roi

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        flat = norm[roi].reshape(-1, 1)
        km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0,
                             batch_size=4096, max_iter=100).fit(flat)
        labels_full = np.full(norm.shape, -1, dtype=np.int32)
        labels_full[roi] = km.predict(norm[roi].reshape(-1, 1))
        means = [norm[labels_full == i].mean() if (labels_full == i).any() else -1
                 for i in range(k)]
        return (labels_full == int(np.argmax(means))) & roi

    if method == "regiongrow":
        sm = ndimage.gaussian_filter(norm, 1.0)          # 先平滑，抑制噪声种子
        if seed is None:
            tmp = np.where(roi, sm, -np.inf)
            seed = tuple(int(v) for v in np.unravel_index(np.argmax(tmp), sm.shape))
        tol = float(tol)
        seed_val = sm[seed]
        grown = np.zeros_like(sm, dtype=bool)
        grown[seed] = True
        for _ in range(int(max(volume.shape)) * 2):
            dil = ndimage.binary_dilation(grown, iterations=1)
            new = dil & (np.abs(sm - seed_val) < tol) & roi & ~grown
            if not new.any():
                break
            grown |= new
        return grown

    if method == "watershed":
        sm = ndimage.gaussian_filter(norm, 1.0)
        grad = filters.sobel(sm)
        markers = np.zeros_like(norm, dtype=np.int32)
        markers[~roi] = 1                                       # 背景
        markers[roi & (sm > np.percentile(sm[roi], 95))] = 2     # 确定前景
        markers[roi & (sm < np.percentile(sm[roi], 70))] = 1     # 确定背景
        ws = skseg.watershed(grad, markers, mask=roi)
        return ws == 2

    raise ValueError(f"未知分割方法: {method}")


# ----------------------------------------------------------------------
# 评价指标（对应论文里的 Dice / HD95 / 灵敏度 / 特异度）
# ----------------------------------------------------------------------
def _surface(mask: np.ndarray) -> np.ndarray:
    er = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~er


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    denom = pred.sum() + truth.sum()
    return float(2 * inter / denom) if denom else 1.0


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = np.logical_or(pred, truth).sum()
    inter = np.logical_and(pred, truth).sum()
    return float(inter / union) if union else 1.0


def hausdorff(pred: np.ndarray, truth: np.ndarray,
              spacing: tuple[float, float, float] = (1, 1, 1),
              percentile: float = 95.0) -> float:
    """HD95 / mHD：预测边界到真实边界、以及反向的距离分布统计（mm）。"""
    if pred.sum() == 0 or truth.sum() == 0:
        return float("nan")
    sp = _surface(pred)
    st = _surface(truth)
    dt_truth = ndimage.distance_transform_edt(~st, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d1 = dt_truth[sp]
    d2 = dt_pred[st]
    both = np.concatenate([d1, d2])
    return float(np.percentile(both, percentile))


def evaluate(pred: np.ndarray, truth: np.ndarray,
             spacing: tuple[float, float, float] = (1, 1, 1)) -> dict:
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "Dice": round(dice(pred, truth), 4),
        "IoU": round(iou(pred, truth), 4),
        "HD95 (mm)": round(hausdorff(pred, truth, spacing, 95), 3),
        "mHD (mm)": round(hausdorff(pred, truth, spacing, 50), 3),
        "灵敏度": round(float(sens), 4),
        "特异度": round(float(spec), 4),
        "准确率": round(float(acc), 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ----------------------------------------------------------------------
# 不确定性量化（SPU-Net 范式）
# ----------------------------------------------------------------------
@dataclass
class UQResult:
    mean_prob: np.ndarray      # 平均预测（连续值 0~1）
    variance: np.ndarray       # 预测方差 = 体素级不确定性
    entropy: np.ndarray        # 信息熵（脑膜瘤论文用法）
    members: list[np.ndarray]  # 每次扰动的二值结果
    consensus: np.ndarray      # 用 Otsu 聚合后的最终分割
    truth: np.ndarray | None = None


def _random_rotation(vol: np.ndarray, angle: float, axes: tuple[int, int]) -> np.ndarray:
    return ndimage.rotate(vol, angle, axes=axes, reshape=False, order=1, mode="nearest")


def perturbation_uncertainty(
    volume: np.ndarray,
    method: str = "otsu",
    n_views: int = 8,
    max_angle: float = 12.0,
    noise: float = 0.0,
    seed: int = 0,
    **seg_kwargs,
) -> UQResult:
    """多视角扰动 → 多组分割 → 方差/熵作为不确定性（SPU-Net 的核心机制）。"""
    rng = np.random.RandomState(seed)
    members: list[np.ndarray] = []
    axes_pool = [(0, 1), (0, 2), (1, 2)]

    for i in range(n_views):
        if i == 0:
            vol_i = volume.copy()                       # 第 0 次为原始视角
        else:
            ang = rng.uniform(-max_angle, max_angle)
            ax = axes_pool[rng.randint(len(axes_pool))]
            vol_i = _random_rotation(volume.astype(np.float32), ang, ax)
            if noise > 0:
                vol_i = vol_i + rng.randn(*volume.shape) * noise

        m = segment(vol_i, method=method, **seg_kwargs)
        if i != 0:
            # 反向旋转回原空间（用最近邻，保持标签）
            ang_back = -ang
            m = ndimage.rotate(m.astype(np.uint8), ang_back, axes=ax,
                               reshape=False, order=0, mode="nearest").astype(bool)
        members.append(m)

    stack = np.stack([m.astype(np.float32) for m in members], axis=0)
    mean_prob = stack.mean(axis=0)
    variance = stack.var(axis=0)

    # 信息熵：用 scipy.special.entr（entr(0)=0），
    # 这样「所有视角结论一致」的体素熵恰好为 0 —— 后续分析才能只在真正
    # 有分歧的体素上做分箱，否则分位数会被大片 0 压塌。
    from scipy.special import entr
    entropy = (entr(mean_prob) + entr(1.0 - mean_prob)) / np.log(2)   # 以 2 为底

    consensus = otsu_aggregate(mean_prob)
    return UQResult(mean_prob=mean_prob, variance=variance, entropy=entropy,
                    members=members, consensus=consensus)


def otsu_aggregate(prob: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    """用 Otsu 法把概率图二值化（对应原论文的聚合步骤）。"""
    from skimage import filters
    vals = prob if roi is None else prob[roi]
    if vals.size == 0 or vals.max() <= vals.min():
        return prob > 0.5
    t = filters.threshold_otsu(vals)
    return prob > t


def uncertainty_groups(uq: UQResult, truth: np.ndarray) -> list[dict]:
    """两组对比：所有视角「一致」的体素 vs 「有分歧」的体素，各自的错误率。

    这是不确定性量化最直接、最稳健的证据形式（不受分箱方式影响）。
    """
    err = (uq.consensus != truth)
    disagree = uq.entropy > 1e-9          # 12 次扰动里有分歧
    agree = ~disagree
    rows = []
    for label, sel in [("一致体素（熵 = 0）", agree), ("有分歧体素（熵 > 0）", disagree)]:
        if sel.sum() == 0:
            continue
        rows.append({
            "分组": label,
            "体素数": int(sel.sum()),
            "占比": f"{100*sel.mean():.1f}%",
            "错误体素数": int(err[sel].sum()),
            "错误率": round(float(err[sel].mean()), 4),
        })
    return rows


def uncertainty_vs_error(uq: UQResult, truth: np.ndarray,
                         n_bins: int = 5, measure: str = "entropy") -> list[dict]:
    """检验「高不确定区域是否真的更容易出错」——不确定性量化的核心命题。

    measure：
      - "entropy"（默认）：连续的信息熵，分箱稳定，推荐
      - "variance"：预测方差（多视角时取值离散，可能只有 2–3 个不同值）

    用**分位数分箱**而非等宽分箱，保证退化情形也能出结果。
    """
    err = (uq.consensus != truth)
    unc = uq.entropy if measure == "entropy" else uq.variance

    # 关键：只在「存在不确定性」的体素里比较。
    # 远离边界的体素在 12 次扰动下结论完全一致（熵=0），把它们算进来会把分箱压塌。
    pos = unc > 1e-9
    if pos.sum() < 50:
        return [{
            "不确定性分位": "全部体素（不确定性均为 0）",
            "体素数": int(unc.size),
            "错误体素数": int(err.sum()),
            "错误率": round(float(err.mean()), 4),
            "平均不确定性": round(float(unc.mean()), 4),
        }]

    vals = unc[pos]
    errs = err[pos]

    edges = np.unique(np.percentile(vals, np.linspace(0, 100, n_bins + 1)))
    if edges.size < 2:
        return [{
            "不确定性分位": "全部（取值退化）",
            "体素数": int(vals.size),
            "错误体素数": int(errs.sum()),
            "错误率": round(float(errs.mean()), 4),
            "平均不确定性": round(float(vals.mean()), 4),
        }]

    rows = []
    n = len(edges) - 1
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        sel = (vals >= lo) & (vals <= hi) if i == n - 1 else (vals >= lo) & (vals < hi)
        if sel.sum() == 0:
            continue
        rows.append({
            "不确定性分位": f"第 {i+1} 档 / {n}" if n <= 3 else f"{int(i*100/n)}–{int((i+1)*100/n)}%",
            "体素数": int(sel.sum()),
            "错误体素数": int(errs[sel].sum()),
            "错误率": round(float(errs[sel].mean()), 4),
            "平均不确定性": round(float(vals[sel].mean()), 4),
        })
    return rows
