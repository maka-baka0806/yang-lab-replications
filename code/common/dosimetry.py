"""
放疗剂量学工具（Dosimetry）
=============================
对应文献：
  - Yang Z, et al. Total brain dose estimation in single-isocenter-multiple-targets
    (SIMT) radiosurgery via a novel deep neural network with spherical convolutions.
    Med Phys 2025;52(6):4266-4277.（球形投影 + 球面卷积预测 V10Gy/V12Gy）
  - Li Z†, Chen K†, Yang Z, et al. A personalized DVH prediction model for HDR
    brachytherapy in cervical cancer treatment. Front Oncol 2022;12:967436.
  - Shu X, ..., Yang Z, et al. Knowledge-based DRU for synthetic CT generation ...
    J Appl Clin Med Phys 2026.（Gamma 指数验证）

本模块提供三件临床科研的基础工具：
  1. DVH 计算与剂量指标（D2cc / Dmean / V10Gy / V12Gy …）
  2. Gamma 指数（3%/2mm 标准，TG-218）
  3. 球形投影（把 3D 靶区分布压成 2D 球面图 —— SCNN 论文的核心变换）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------
# 1. 合成剂量分布与 SIMT 体模
# ----------------------------------------------------------------------
@dataclass
class DoseCase:
    dose: np.ndarray           # 剂量分布（Gy）
    targets: np.ndarray        # 靶区掩膜
    brain: np.ndarray          # 正常脑组织掩膜
    prescription: float        # 处方剂量（Gy）
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_targets: int = 0


def make_simt_case(size: int = 64, n_targets: int = 4, prescription: float = 20.0,
                   seed: int = 0, target_radius: int = 4) -> DoseCase:
    """模拟一个「单等中心多靶点」脑放疗病例。

    剂量模型：每个靶区贡献一个「平台 + 边缘跌落」的剂量核（近似 SRS 的陡峭剂量梯度），
    总剂量为各靶区贡献的叠加，并限制在脑组织内。
    """
    rng = np.random.RandomState(seed)
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)

    brain = r <= size * 0.42
    targets = np.zeros((size, size, size), dtype=bool)

    centers = []
    for _ in range(n_targets):
        for _try in range(50):
            p = rng.uniform(-0.30, 0.30, size=3) * size
            if np.linalg.norm(p) + target_radius < size * 0.38 and \
               all(np.linalg.norm(p - q) > target_radius * 2.2 for q in centers):
                centers.append(p)
                break
    for p in centers:
        d = np.sqrt((xx - c - p[0]) ** 2 + (yy - c - p[1]) ** 2 + (zz - c - p[2]) ** 2)
        targets |= d <= target_radius

    # 剂量核：靶区内为处方剂量，边缘按误差函数跌落
    dose = np.zeros((size, size, size), dtype=np.float32)
    for p in centers:
        d = np.sqrt((xx - c - p[0]) ** 2 + (yy - c - p[1]) ** 2 + (zz - c - p[2]) ** 2)
        from scipy.special import erfc
        falloff = 0.5 * erfc((d - target_radius) / 2.0)
        dose += prescription * falloff

    dose = np.clip(dose, 0, None) * brain

    return DoseCase(dose=dose, targets=targets, brain=brain,
                    prescription=prescription, n_targets=len(centers))


# ----------------------------------------------------------------------
# 2. DVH 与剂量指标
# ----------------------------------------------------------------------
def dvh_curve(dose: np.ndarray, structure: np.ndarray,
              n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """累积 DVH：横轴剂量（Gy），纵轴「接受 ≥ 该剂量的体积百分比」。"""
    if structure.sum() == 0:
        return np.array([0.0]), np.array([0.0])
    d = dose[structure]
    dmax = float(d.max()) if d.max() > 0 else 1.0
    grid = np.linspace(0, dmax, n_bins)
    cum = np.array([(d >= g).mean() * 100 for g in grid])
    return grid, cum


def dose_metrics(dose: np.ndarray, structure: np.ndarray,
                 spacing: tuple[float, float, float] = (1, 1, 1),
                 v_levels: tuple[float, ...] = (10.0, 12.0)) -> dict:
    """计算常用剂量指标。

    D2cc / D1cc / D0.1cc：最热 2cc / 1cc / 0.1cc 体积所接受的最小剂量（近距离治疗常用）
    Dmax / Dmean / D95：最大 / 平均 / 95% 体积接受的剂量
    V10Gy / V12Gy：接受 ≥10 Gy / ≥12 Gy 的体积（cc）
    """
    if structure.sum() == 0:
        return {}
    d = dose[structure]
    voxel_cc = float(np.prod(spacing)) / 1000.0     # mm³ → cc
    n = d.size
    d_sorted = np.sort(d)[::-1]

    def d_x_cc(x_cc: float) -> float:
        k = int(np.clip(round(x_cc / voxel_cc), 1, n))
        return float(d_sorted[k - 1])

    out = {
        "体积 (cc)": round(n * voxel_cc, 2),
        "Dmean (Gy)": round(float(d.mean()), 3),
        "Dmax (Gy)": round(float(d.max()), 3),
        "D2cc (Gy)": round(d_x_cc(2.0), 3),
        "D1cc (Gy)": round(d_x_cc(1.0), 3),
        "D0.1cc (Gy)": round(d_x_cc(0.1), 3),
    }
    for v in v_levels:
        out[f"V{v:g}Gy (cc)"] = round(float((d >= v).sum() * voxel_cc), 3)
        out[f"V{v:g}Gy (%)"] = round(float((d >= v).mean() * 100), 2)
    return out


def simt_metrics(case: DoseCase) -> dict:
    """SIMT 病例的关键指标：正常脑受照体积（对应 SCNN 论文的预测目标）。"""
    m = dose_metrics(case.dose, case.brain, case.spacing, v_levels=(10.0, 12.0))
    pres = case.prescription
    brain = case.brain
    voxel_cc = float(np.prod(case.spacing)) / 1000.0
    out = {
        "靶点数": case.n_targets,
        "靶区总体积 (cc)": round(float(case.targets.sum()) * voxel_cc, 2),
        "脑体积 (cc)": round(float(brain.sum()) * voxel_cc, 2),
    }
    for pct in (50, 60, 66.7):
        lvl = pres * pct / 100
        out[f"V{pct:g}% (cc)"] = round(float(((case.dose >= lvl) & brain).sum() * voxel_cc), 3)
    out["V10Gy (cc)"] = m.get("V10Gy (cc)")
    out["V12Gy (cc)"] = m.get("V12Gy (cc)")
    return out


# ----------------------------------------------------------------------
# 3. Gamma 指数（TG-218：3%/2mm）
# ----------------------------------------------------------------------
def _shift3(arr: np.ndarray, d: tuple[int, int, int]) -> np.ndarray:
    """按 (dz, dy, dx) 平移数组，边界补零。"""
    out = np.zeros_like(arr)
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    for ax, s in enumerate(d):
        if s > 0:
            src[ax] = slice(0, -s); dst[ax] = slice(s, None)
        elif s < 0:
            src[ax] = slice(-s, None); dst[ax] = slice(0, s)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def gamma_index(dose_ref: np.ndarray, dose_eval: np.ndarray,
                spacing: tuple[float, float, float] = (1, 1, 1),
                dose_tol_pct: float = 3.0, dist_tol_mm: float = 2.0,
                threshold_pct: float = 10.0) -> dict:
    """计算 Gamma 通过率（全局归一化，标准 3%/2mm）。

    返回通过率、Gamma 图，以及低剂量阈值以上的体素数。
    """
    dmax = float(dose_ref.max())
    if dmax <= 0:
        return {"通过率": float("nan"), "gamma": np.zeros_like(dose_ref), "评估体素数": 0}

    dd = dose_tol_pct / 100.0 * dmax
    R = int(np.ceil(dist_tol_mm / min(spacing))) + 1

    best = np.full(dose_ref.shape, np.inf, dtype=np.float32)
    for dz in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                dist2 = (dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2 + (dx * spacing[2]) ** 2
                if dist2 > (dist_tol_mm + 1e-9) ** 2:
                    continue
                shifted = _shift3(dose_eval, (dz, dy, dx))
                diff = (dose_ref - shifted) / dd
                g2 = dist2 / (dist_tol_mm ** 2) + diff ** 2
                np.minimum(best, g2, out=best)

    gamma = np.sqrt(best)
    mask = dose_ref >= threshold_pct / 100.0 * dmax
    passing = float((gamma[mask] <= 1.0).mean()) if mask.sum() else float("nan")
    return {
        "通过率": passing,
        "gamma": gamma,
        "评估体素数": int(mask.sum()),
        "评估掩膜": mask,
        "最大 gamma": float(gamma[mask].max()) if mask.sum() else float("nan"),
    }


# ----------------------------------------------------------------------
# 4. 球形投影（SCNN 论文的核心变换）
# ----------------------------------------------------------------------
@dataclass
class SphereProjection:
    density: np.ndarray        # (n_phi, n_theta) 投影密度（2D 球面图）
    n_theta: int
    n_phi: int
    coverage: float            # 靶区在球面上占据的角度比例
    peak: float                # 最大角向密度
    centroid_rms: float        # 径向集中度（靶区离等中心多远）


def spherical_projection(mask: np.ndarray,
                         n_theta: int = 48, n_phi: int = 24) -> SphereProjection:
    """把 3D 靶区分布投影到球面（方位角 × 极角）。

    这正是 SCNN 论文的做法：脑被"装进"一个球，靶区体素用球坐标表示，
    于是 3D 几何问题变成 2D 球面图像问题 —— 参数量因此减少一个数量级。
    """
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return SphereProjection(np.zeros((n_phi, n_theta)), n_theta, n_phi, 0, 0, 0)

    center = np.array(mask.shape) / 2.0
    rel = idx - center
    z, y, x = rel[:, 0], rel[:, 1], rel[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-9
    theta = np.arctan2(y, x)                       # 方位角 [-π, π]
    phi = np.arccos(np.clip(z / r, -1, 1))         # 极角 [0, π]

    hist, _, _ = np.histogram2d(phi, theta, bins=[n_phi, n_theta],
                                range=[[0, np.pi], [-np.pi, np.pi]])
    occupied = (hist > 0)
    coverage = float(occupied.mean())
    peak = float(hist.max())
    centroid_rms = float(np.sqrt((r ** 2).mean()))

    return SphereProjection(density=hist, n_theta=n_theta, n_phi=n_phi,
                            coverage=coverage, peak=peak, centroid_rms=centroid_rms)


def sphere_feature_vector(mask: np.ndarray, n_theta: int = 48,
                          n_phi: int = 24) -> dict:
    """从球形投影提取几何特征（SCNN 实际用的是整张球面图，这里给可解释的汇总量）。"""
    sp = spherical_projection(mask, n_theta, n_phi)
    d = sp.density
    tot = d.sum() + 1e-9
    p = d / tot
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "角向覆盖率": round(sp.coverage, 4),
        "角向峰值密度": round(sp.peak, 1),
        "角向熵": round(entropy, 4),
        "径向集中度 (体素)": round(sp.centroid_rms, 2),
        "靶区体素数": int(mask.sum()),
    }
