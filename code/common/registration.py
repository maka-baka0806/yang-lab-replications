"""
形变配准与物理合理性评估（Image Registration）
================================================
对应文献：
  - Zhang Z, Guo D, ..., Yang Z. PhysMorph: A biomechanical and image-guided deep
    learning framework for real-time multi-modal liver image registration.
    Phys Imaging Radiat Oncol 2026;37:100906.
  - Sun P, Zhang C, Yang Z, et al. An Implicit Registration Framework Integrating
    Kolmogorov-Arnold Networks with Velocity Regularization for IGRT.
    Bioengineering 2025;12(9):1005.

PhysMorph 的三个关键动作（本模块全部实现）：
  1. 用现成的配准算法（此处为 SimpleITK 的 Demons）估计形变场 DVF
  2. 用 **Jacobian 行列式** 检查形变是否物理合理（负值 = 组织自折叠，不允许）
  3. 用 **TRE**（标志点误差）与 **MSD**（表面平均距离）量化精度，
     并对比「传统迭代法」与「快速方法」的时间代价
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


# ----------------------------------------------------------------------
# 合成形变体模：已知真值形变场，可精确计算 TRE
# ----------------------------------------------------------------------
@dataclass
class RegistrationCase:
    fixed: np.ndarray          # 参考图像（治疗前，如 MRI）
    moving: np.ndarray         # 待配准图像（治疗中，如 CBCT）
    organ: np.ndarray          # 器官掩膜
    dvf_true: np.ndarray       # 真值位移场 (3, z, y, x)
    landmarks_fixed: np.ndarray
    landmarks_moving: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


def make_registration_case(size: int = 56, magnitude: float = 4.0,
                           n_landmarks: int = 12, seed: int = 0) -> RegistrationCase:
    """构造一个「器官 + 内部结构」的合成病例，并施加已知的平滑形变。"""
    rng = np.random.RandomState(seed)
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]

    organ = (((xx - c) / (size * 0.34)) ** 2 +
             ((yy - c) / (size * 0.28)) ** 2 +
             ((zz - c) / (size * 0.30)) ** 2) <= 1

    # 参考图像：器官内带纹理 + 几个内部结构（模拟血管/病灶）
    fixed = np.zeros((size, size, size), dtype=np.float32)
    texture = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 1.2)
    fixed[organ] = 120 + texture[organ] * 30
    blobs = []
    for _ in range(5):
        p = rng.uniform(0.25, 0.75, size=3) * size
        d = np.sqrt((xx - p[0]) ** 2 + (yy - p[1]) ** 2 + (zz - p[2]) ** 2)
        fixed[d <= 3.5] += 90
        blobs.append(p)

    # 平滑形变场（低频随机场 × 幅度）
    dvf = np.stack([
        ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 6)
        for _ in range(3)
    ])
    for i in range(3):
        dvf[i] = dvf[i] / (np.abs(dvf[i]).max() + 1e-9) * magnitude

    # 用形变场搬移参考图像 → moving
    coords = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    warped_coords = [coords[i] + dvf[i] for i in range(3)]
    moving = ndimage.map_coordinates(fixed, warped_coords, order=1, mode="nearest")

    # 标志点：在器官内部随机取点，按真值场搬移
    lm_fixed = []
    while len(lm_fixed) < n_landmarks:
        p = rng.uniform(0.2, 0.8, size=3) * size
        if organ[tuple(p.astype(int))]:
            lm_fixed.append(p)
    lm_fixed = np.array(lm_fixed)
    lm_moving = np.array([
        [p[i] + float(ndimage.map_coordinates(dvf[i], [[p[0]], [p[1]], [p[2]]],
                                              order=1, mode="nearest")[0])
         for i in range(3)]
        for p in lm_fixed
    ])

    return RegistrationCase(fixed=fixed, moving=moving, organ=organ,
                            dvf_true=dvf, landmarks_fixed=lm_fixed,
                            landmarks_moving=lm_moving)


# ----------------------------------------------------------------------
# 配准（调用 SimpleITK 现成算法）
# ----------------------------------------------------------------------
METHODS = {
    "fast_symmetric": "Fast Symmetric Forces Demons（推荐：快且稳）",
    "demons": "Demons 形变配准（经典快速）",
}


def register(case: RegistrationCase, method: str = "fast_symmetric",
             iterations: int = 60) -> dict:
    """执行配准，返回形变场与耗时。"""
    import time

    fixed = sitk.GetImageFromArray(case.fixed.astype(np.float32))
    moving = sitk.GetImageFromArray(case.moving.astype(np.float32))
    fixed.SetSpacing(case.spacing)
    moving.SetSpacing(case.spacing)

    # 统一强度范围，避免 Demons 对强度尺度敏感
    fixed = sitk.Normalize(fixed)
    moving = sitk.Normalize(moving)

    t0 = time.time()
    if method in ("demons", "fast_symmetric"):
        if method == "demons":
            reg = sitk.DemonsRegistrationFilter()
        else:
            reg = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        reg.SetNumberOfIterations(int(iterations))
        reg.SetStandardDeviations(1.0)
        dvf_img = reg.Execute(fixed, moving)
        dvf = sitk.GetArrayFromImage(dvf_img)          # SimpleITK 返回 (z, y, x, 3)
        if dvf.shape[-1] == 3:
            dvf = np.transpose(dvf, (3, 0, 1, 2))      # → (3, z, y, x)
    elif method == "bspline":
        tx = sitk.BSplineTransformInitializer(fixed, [8, 8, 8])
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                               numberOfIterations=int(iterations))
        R.SetInitialTransform(tx, inPlace=True)
        R.SetInterpolator(sitk.sitkLinear)
        out_tx = R.Execute(fixed, moving)
        # 由变换场生成 DVF
        field = sitk.TransformToDisplacementField(
            out_tx, sitk.sitkVectorFloat64, fixed.GetSize(),
            fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())
        dvf = sitk.GetArrayFromImage(field)
        dvf = np.transpose(dvf, (3, 0, 1, 2))          # (z,y,x,3) → (3,z,y,x)
    else:
        raise ValueError(method)

    elapsed = time.time() - t0

    # 用估计出的 DVF 把 moving 重采样回 fixed 空间，得到配准后图像
    warped = _apply_dvf(case.moving, dvf)
    return {"dvf": dvf, "warped": warped, "耗时 (s)": round(elapsed, 3),
            "method": method}


def _apply_dvf(image: np.ndarray, dvf: np.ndarray) -> np.ndarray:
    """按位移场搬移图像（用于显示配准结果）。"""
    size = image.shape
    coords = np.mgrid[0:size[0], 0:size[1], 0:size[2]].astype(np.float32)
    warped_coords = [coords[i] + dvf[i] for i in range(3)]
    return ndimage.map_coordinates(image, warped_coords, order=1, mode="nearest")


# ----------------------------------------------------------------------
# 物理合理性：Jacobian 行列式（PhysMorph 的关键质控）
# ----------------------------------------------------------------------
def jacobian_determinant(dvf: np.ndarray) -> np.ndarray:
    """计算形变场的 Jacobian 行列式。

    det(J) > 1：局部膨胀；0 < det(J) < 1：局部压缩；
    **det(J) ≤ 0：组织自我折叠 —— 物理上不可能，配准失败的标志。**
    """
    # 对每个分量求三个方向的偏导（中心差分）
    grads = []
    for i in range(3):
        g = np.gradient(dvf[i])
        grads.append(g)                      # 每个 g 是 [d/dz, d/dy, d/dx]
    # J[i, j] = ∂u_i/∂x_j；单位阵 + 位移梯度
    J = np.empty(dvf.shape[1:] + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            J[..., i, j] = grads[i][j]
    for i in range(3):
        J[..., i, i] += 1.0
    det = np.linalg.det(J)
    return det


def folding_stats(dvf: np.ndarray, organ: np.ndarray | None = None) -> dict:
    """统计形变的物理合理性（对应 PhysMorph 论文报告的指标）。"""
    det = jacobian_determinant(dvf)
    vals = det[organ] if organ is not None else det.ravel()
    neg = float((vals <= 0).mean())
    return {
        "负 Jacobian 比例": round(neg, 6),
        "折叠体素数": int((vals <= 0).sum()),
        "det 均值": round(float(vals.mean()), 4),
        "det 标准差": round(float(vals.std()), 4),
        "最差 det": round(float(vals.min()), 4),
        "判断": "✅ 物理合理" if neg < 1e-4 else ("⚠️ 存在轻微折叠" if neg < 1e-2 else "❌ 严重折叠"),
    }


# ----------------------------------------------------------------------
# 精度评估：TRE / MSD
# ----------------------------------------------------------------------
def target_registration_error(dvf: np.ndarray, case: RegistrationCase) -> float:
    """标志点误差（mm）：真值标志点位置 vs 由估计形变场预测的位置。"""
    errs = []
    for p_fix, p_mov in zip(case.landmarks_fixed, case.landmarks_moving):
        pred = []
        for i in range(3):
            v = ndimage.map_coordinates(dvf[i], [[p_mov[0]], [p_mov[1]], [p_mov[2]]],
                                        order=1, mode="nearest")[0]
            pred.append(p_mov[i] + float(v))
        pred = np.array(pred)
        errs.append(np.linalg.norm(pred - p_fix))
    return float(np.mean(errs))


def mean_surface_distance(a: np.ndarray, b: np.ndarray,
                          spacing: tuple[float, float, float] = (1, 1, 1)) -> float:
    """两个掩膜表面的平均距离（mm）。"""
    def surf(m):
        return m & ~ndimage.binary_erosion(m, iterations=1)
    sa, sb = surf(a), surf(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return float("nan")
    da = ndimage.distance_transform_edt(~sa, sampling=spacing)
    db = ndimage.distance_transform_edt(~sb, sampling=spacing)
    return float((da[sb].mean() + db[sa].mean()) / 2)


def evaluate_registration(result: dict, case: RegistrationCase) -> dict:
    """综合评估：精度（TRE / MSD）+ 物理合理性（Jacobian）+ 耗时。"""
    dvf = result["dvf"]
    tre = target_registration_error(dvf, case)
    # 配准后器官掩膜（用估计形变搬移 moving 的器官）
    moving_organ = _apply_dvf(case.organ.astype(np.float32), dvf) > 0.5
    msd = mean_surface_distance(moving_organ, case.organ.astype(bool), case.spacing)

    # 形变场本身的误差（相对真值）
    dvf_err = float(np.sqrt(((dvf - case.dvf_true) ** 2).sum(axis=0).mean()))

    return {
        "方法": METHODS.get(result["method"], result["method"]),
        "TRE (mm)": round(tre, 3),
        "MSD (mm)": round(msd, 3),
        "DVF 端点误差 (mm)": round(dvf_err, 3),
        "耗时 (s)": result["耗时 (s)"],
        **folding_stats(dvf, case.organ),
    }
