"""
4DCT 体素级时序放射组学（Time-series Radiomics）
==================================================
对应文献：
  - Zhang R, …, Yang Z. An Explainable Neural Radiomic Sequence Model with
    Spatiotemporal Continuity for Quantifying 4DCT-based Pulmonary Ventilation.
    arXiv:2503.23898（2025）

论文核心（本模块逐条复现）：
  1. 把静态 CT 升级为 **4DCT 的呼吸周期序列**
  2. 在每个体素、每个呼吸相位上提取特征 → 构成「放射组学序列」
  3. 用时序模型判断该体素是否为通气缺陷
  4. 生成**时间显著性图**，找出关键相位与关键特征
  5. 论文结论：呼气相时，功能受损区表现为
     **(a) 强度上升趋势、(b) 均匀性下降趋势** —— 与健康肺相反
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Lung4D:
    volume: np.ndarray        # (n_phases, z, y, x) 4DCT 序列（HU）
    mask: np.ndarray          # 肺掩膜
    defect: np.ndarray        # 真值通气缺损区
    reference: np.ndarray     # 参考通气图（相当于 Galligas PET / DTPA SPECT）
    phase_names: list[str]
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _phase_names(n: int) -> list[str]:
    """0% = 吸气末，50% = 呼气末（与临床 4DCT 相位惯例一致）。"""
    return [f"{int(round(100 * i / n))}%" for i in range(n)]


def make_4dct(size: int = 48, n_phases: int = 10, defect_side: str = "left",
              defect_radius: int = 7, noise: float = 8.0, seed: int = 0) -> Lung4D:
    """构造带「气体潴留」缺损的合成 4DCT。

    生理模型：
      - 健康肺：吸气时扩张、密度下降；呼气时回缩、密度上升（但幅度有限）
      - 缺损区（气体潴留 air trapping）：呼气时**不能排空**，
        因此密度上升幅度更大，且纹理变得更不均匀
    """
    rng = np.random.RandomState(seed)
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]

    # 缺损区位置
    dc = (c - 11, c, c) if defect_side == "left" else (c + 11, c, c)
    defect_full = ((xx - dc[0]) ** 2 + (yy - dc[1]) ** 2 + (zz - dc[2]) ** 2) <= defect_radius ** 2

    phases = []
    for p in range(n_phases):
        t = p / n_phases                      # 0 → 1 一个呼吸周期
        breath = np.sin(2 * np.pi * t)         # +1 吸气末 … −1 呼气末
        # 肺随呼吸缩放（膈肌运动）
        scale_z = 1.0 + 0.14 * breath
        scale_xy = 1.0 + 0.07 * breath

        left = (((xx - c + 11) / (9 * scale_xy)) ** 2 +
                ((yy - c) / (11 * scale_xy)) ** 2 +
                ((zz - c) / (13 * scale_z)) ** 2) <= 1
        right = (((xx - c - 11) / (9 * scale_xy)) ** 2 +
                 ((yy - c) / (11 * scale_xy)) ** 2 +
                 ((zz - c) / (13 * scale_z)) ** 2) <= 1
        lung = left | right
        if p == 0:
            first_lung = lung
        defect = defect_full & lung

        # ---- 健康肺：密度随呼吸变化（吸气变低、呼气变高）----
        base = -880.0 + 60.0 * (1 - breath) / 2      # −880 → −820
        tex = rng.randn(size, size, size) * noise
        low = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 4) * 40

        vol = np.full((size, size, size), -1000.0, dtype=np.float32)
        vol[lung] = base + tex[lung] + low[lung]

        # ---- 缺损区：呼气时密度升高（气体排不出），且纹理随呼气变得更粗糙 ----
        # 论文结论：受损区在呼气相「强度上升 + 均匀性下降」，与健康肺相反
        trap = -810.0 + 95.0 * (1 - breath) / 2
        exhale_amt = (1 - breath) / 2                # 0（吸气末）→ 1（呼气末）
        rough_sd = noise * (1.3 + 2.2 * exhale_amt)  # 越到呼气末越粗糙
        rough = rng.randn(size, size, size) * rough_sd
        vol[defect] = trap + rough[defect] + low[defect] * 2.2

        phases.append(vol)

    volume = np.stack(phases, axis=0)

    # 参考通气图：健康 = 1，缺损 = 低
    ref = np.zeros((size, size, size), dtype=np.float32)
    ref[first_lung] = 1.0
    ref[defect_full & first_lung] = 0.15
    ref = ndimage.gaussian_filter(ref, 2.0)

    return Lung4D(volume=volume, mask=first_lung, defect=defect_full & first_lung,
                  reference=np.where(first_lung, ref, np.nan),
                  phase_names=_phase_names(n_phases))


# ----------------------------------------------------------------------
# 体素级特征序列
# ----------------------------------------------------------------------
def voxel_feature_series(vol4d: np.ndarray, mask: np.ndarray,
                         kernel: int = 5) -> dict[str, np.ndarray]:
    """在每个呼吸相位上计算体素级特征，返回形状 (n_phases, z, y, x) 的序列。

    - intensity：局部平均强度（对应论文的强度特征）
    - homogeneity：局部同质性 1/(1+方差)（对应论文的均匀性特征）
    """
    n = vol4d.shape[0]
    inten = np.empty_like(vol4d)
    homo = np.empty_like(vol4d)
    size = (kernel,) * 3

    for p in range(n):
        v = vol4d[p].astype(np.float32)
        mean = ndimage.uniform_filter(v, size=size, mode="nearest")
        mean_sq = ndimage.uniform_filter(v ** 2, size=size, mode="nearest")
        var = np.maximum(mean_sq - mean ** 2, 0)
        inten[p] = mean
        homo[p] = 1.0 / (1.0 + var / 400.0)      # 归一化到 0–1

    return {"intensity": inten, "homogeneity": homo}


# ----------------------------------------------------------------------
# 时序特征 → 缺陷判别
# ----------------------------------------------------------------------
def series_features(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把每个体素的时间序列压成特征向量：斜率、极差、呼气/吸气比、末段变化。

    返回 shape (n_voxels_in_mask, n_features)
    """
    ts = series[:, mask]                        # (n_phases, n_voxels)
    n = ts.shape[0]
    t = np.linspace(0, 1, n)

    slope = np.array([np.polyfit(t, ts[:, i], 1)[0] for i in range(ts.shape[1])]) \
        if ts.shape[1] < 3000 else _slope_fast(ts, t)
    rng_ = ts.max(axis=0) - ts.min(axis=0)
    exhale = ts[n // 2:].mean(axis=0)           # 呼气相（后半段）
    inhale = ts[:max(n // 4, 1)].mean(axis=0)   # 吸气末附近
    ratio = exhale - inhale
    late = ts[-1] - ts[n // 2]

    return np.column_stack([slope, rng_, ratio, late])


def _slope_fast(ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """向量化的最小二乘斜率（体素多时用）。"""
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    return (t_c[:, None] * (ts - ts.mean(axis=0))).sum(axis=0) / denom


def classify_voxels(feats: np.ndarray, mask: np.ndarray,
                    truth: np.ndarray) -> dict:
    """用逻辑回归把体素分成「缺陷 / 健康」，并给出评价指标。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    y = truth[mask].astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"说明": "该体模没有缺损或全部为缺损，无法训练"}
    if feats.shape[1] > 4:
        feats = feats[:, :4]

    sc = StandardScaler().fit(feats)
    Z = sc.transform(feats)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z, y)
    prob = lr.predict_proba(Z)[:, 1]

    from sklearn.metrics import roc_auc_score as auc
    best_dice, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        pred = prob >= thr
        d = 2 * (pred & y).sum() / max(pred.sum() + y.sum(), 1)
        if d > best_dice:
            best_dice, best_thr = float(d), float(thr)

    return {
        "体素数": int(mask.sum()),
        "真值缺损体素": int(y.sum()),
        "AUC": round(float(auc(y, prob)), 4),
        "最佳 Dice": round(best_dice, 4),
        "最佳阈值": round(best_thr, 3),
        "系数（强度斜率等）": np.round(lr.coef_[0], 3).tolist(),
        "_prob": prob, "_y": y, "_thr": best_thr,
    }


# ----------------------------------------------------------------------
# 时间显著性（对应论文的 temporal saliency）
# ----------------------------------------------------------------------
def temporal_saliency(series: np.ndarray, mask: np.ndarray,
                      truth: np.ndarray) -> list[dict]:
    """逐相位计算该时刻特征对「是否为缺陷」的判别力（AUC），形成时间显著性。"""
    from sklearn.metrics import roc_auc_score

    y = truth[mask].astype(int)
    rows = []
    for p in range(series.shape[0]):
        vals = series[p][mask]
        a = roc_auc_score(y, vals)
        rows.append({
            "相位": p,
            "该相位强度的判别 AUC": round(float(max(a, 1 - a)), 4),
            "方向": "值越高越像缺损" if a >= 0.5 else "值越低越像缺损",
        })
    return rows


def group_curves(series: np.ndarray, mask: np.ndarray,
                 truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回（缺损组平均曲线, 健康组平均曲线），对应论文的核心图。"""
    d = truth[mask]
    m = series[:, mask]
    return m[:, d].mean(axis=1), m[:, ~d].mean(axis=1)
