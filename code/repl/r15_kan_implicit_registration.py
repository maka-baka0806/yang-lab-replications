"""
R15 · KAN 隐式配准：用 Jacobi 多项式网络拟合形变场
=====================================================
原文：Sun P, Zhang C, **Yang Z**, Yin FF, Liu M. An Implicit Registration Framework
      Integrating Kolmogorov–Arnold Networks with Velocity Regularization for
      Image-Guided Radiation Therapy. Bioengineering 2025;12(9):1005.

论文的核心主张
--------------
1. 把 KAN（Kolmogorov–Arnold Network）首次用于医学图像配准：KAN 层输出
   Σ_j φ_j(x_j)，每个 φ_j 是可学习的基函数（原文用 Jacobi 多项式构造）；
2. 配准被写成**隐式**形式：形变场不是一张离散网格，而是坐标的连续函数
   u(x, y, z)，任意位置都能解析求值；
3. 对速度场做主成分建模以提速（原文报告约 70%）；
4. 速度正则化保证形变平滑、拓扑保持（Jacobian 不出现负值）。

本复现做什么
------------
用 `registration.make_registration_case` 造一个已知真值形变的合成病例，把
「坐标 → 位移」当成回归任务：
    (a) 坐标 MLP 基线（Tanh 全连接，参数量与 KAN 对齐）
    (b) Jacobi 多项式 KAN（本模块从递推公式手写实现，并与 scipy 对照验证）
    (c) KAN + 速度（位移场梯度）正则化
比较三者的拟合精度、参数量与训练耗时，再用 Jacobian 行列式做物理合理性检查。
"""
from __future__ import annotations

import time
from math import gamma

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.special import eval_jacobi, roots_jacobi

from framework import Step
from common.registration import (make_registration_case, register,
                                 evaluate_registration, folding_stats,
                                 target_registration_error)

# ----------------------------------------------------------------------
# 实验常量
# ----------------------------------------------------------------------
SIZE = 32                 # 体模边长（体素）
MAGNITUDE = 4.0           # 真值形变幅度（体素 = mm）
DEGREE = 4                # Jacobi 多项式最高阶（基函数个数 = DEGREE+1）
ALPHA = BETA = 0.5        # Jacobi 权重参数 (1-x)^α (1+x)^β
KAN_HIDDEN = 13           # KAN 隐层宽度
MLP_HIDDEN = 16           # MLP 隐层宽度（参数量与 KAN 对齐：390 vs 387）
N_TRAIN, N_TEST = 1400, 600
EPOCHS = 300
EPOCHS_LONG = 900
LR = 5e-3
REG_LAMBDAS = (0.0, 0.005, 0.02)   # 速度正则化权重扫描
NOISE_SIGMA = 1.0                  # 带噪实验：位移观测噪声（mm）
N_NOISY = 400                      # 带噪实验：训练点数
COARSE = 16                        # 粗网格边长，用于验证「连续场 + 插值」的提速
MODEL_SEED = 0                     # 模型初始化种子（保证任何调用顺序下结果一致）


# ----------------------------------------------------------------------
# 1. Jacobi 多项式基（numpy 参考实现）
# ----------------------------------------------------------------------
def jacobi_basis_np(x: np.ndarray, degree: int = DEGREE,
                    alpha: float = ALPHA, beta: float = BETA
                    ) -> np.ndarray:
    """归一化 Jacobi 多项式基 {P_n^{(α,β)}/√h_n}，x ∈ [-1, 1]。

    用三点递推实现（numpy.polynomial 没有现成的 Jacobi 类）：
        2(n+1)(n+α+β+1)(2n+α+β) P_{n+1}
            = (2n+α+β+1)[(2n+α+β)(2n+α+β+2)x + α²−β²] P_n
              − 2(n+α)(n+β)(2n+α+β+2) P_{n−1}
    归一化常数 h_n = 2^{α+β+1}/(2n+α+β+1) · Γ(n+α+1)Γ(n+β+1)/(Γ(n+α+β+1) n!)
    使基在权重 (1−x)^α(1+x)^β 下正交归一。
    """
    x = np.asarray(x, dtype=np.float64)
    a, b = float(alpha), float(beta)
    p = [np.ones_like(x), 0.5 * ((a - b) + (a + b + 2.0) * x)]
    for n in range(1, degree):
        A = 2.0 * (n + 1) * (n + a + b + 1) * (2 * n + a + b)
        B = (2 * n + a + b + 1) * ((2 * n + a + b) * (2 * n + a + b + 2) * x
                                   + a * a - b * b)
        C = 2.0 * (n + a) * (n + b) * (2 * n + a + b + 2)
        p.append((B * p[n] - C * p[n - 1]) / A)

    out = []
    for n in range(degree + 1):
        h = (2.0 ** (a + b + 1) / (2 * n + a + b + 1)
             * gamma(n + a + 1) * gamma(n + b + 1)
             / (gamma(n + a + b + 1) * gamma(n + 1)))
        out.append(p[n] / np.sqrt(h))
    return np.stack(out, axis=-1)          # (..., degree+1)


# ----------------------------------------------------------------------
# 2. 同一套基的 PyTorch 实现 + KAN 层
# ----------------------------------------------------------------------
class JacobiBasis(nn.Module):
    """把 numpy 递推搬到 torch，作为 KAN 边缘上的可学习基函数族。"""

    def __init__(self, degree: int = DEGREE, alpha: float = ALPHA,
                 beta: float = BETA):
        super().__init__()
        self.degree = int(degree)
        a, b = float(alpha), float(beta)
        self.a, self.b = a, b
        h = [2.0 ** (a + b + 1) / (2 * n + a + b + 1)
             * gamma(n + a + 1) * gamma(n + b + 1)
             / (gamma(n + a + b + 1) * gamma(n + 1))
             for n in range(degree + 1)]
        self.register_buffer("norm",
                             torch.tensor([1.0 / np.sqrt(v) for v in h],
                                          dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.a, self.b
        p = [torch.ones_like(x), 0.5 * ((a - b) + (a + b + 2.0) * x)]
        for n in range(1, self.degree):
            A = 2.0 * (n + 1) * (n + a + b + 1) * (2 * n + a + b)
            B = (2 * n + a + b + 1) * ((2 * n + a + b) * (2 * n + a + b + 2) * x
                                       + a * a - b * b)
            C = 2.0 * (n + a) * (n + b) * (2 * n + a + b + 2)
            p.append((B * p[n] - C * p[n - 1]) / A)
        return torch.stack([p[n] * self.norm[n] for n in range(self.degree + 1)],
                           dim=-1)        # (..., degree+1)


class JacobiKANLayer(nn.Module):
    """KAN 层：out_o = Σ_j Σ_k c_{o,j,k} · φ_k(x_j)。"""

    def __init__(self, d_in: int, d_out: int, degree: int = DEGREE):
        super().__init__()
        self.basis = JacobiBasis(degree)
        self.coef = nn.Parameter(torch.randn(d_out, d_in, degree + 1) * 0.15)

    @property
    def n_params(self) -> int:
        return self.coef.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.basis(x)                       # (B, d_in, K)
        return torch.einsum("bik,oik->bo", b, self.coef)


class JacobiKAN(nn.Module):
    """两层 KAN：坐标 (3) → 隐层 → 位移 (3)。

    层间用 tanh 把隐层激活压回 [-1, 1] —— Jacobi 基只在 [-1, 1] 上有定义，
    这一步相当于把「样条网格」换成「紧支撑的有界基」。
    """

    def __init__(self, hidden: int = KAN_HIDDEN, degree: int = DEGREE):
        super().__init__()
        torch.manual_seed(MODEL_SEED)          # 让初始化与之前的随机消耗无关
        self.l1 = JacobiKANLayer(3, hidden, degree)
        self.l2 = JacobiKANLayer(hidden, 3, degree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(x)))


def make_mlp(hidden: int = MLP_HIDDEN) -> nn.Module:
    torch.manual_seed(MODEL_SEED)
    return nn.Sequential(
        nn.Linear(3, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 3),
    )


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------
# 3. 采样、训练、评估
# ----------------------------------------------------------------------
def _norm_coords(idx: np.ndarray, size: int) -> torch.Tensor:
    """体素索引 → [-1, 1] 的归一化坐标。"""
    c = 2.0 * idx.astype(np.float32) / (size - 1) - 1.0
    return torch.tensor(c, dtype=torch.float32)


def _smoothness_penalty(model: nn.Module, x: torch.Tensor, d: float) -> torch.Tensor:
    """速度（位移场）正则化：一阶有限差分近似 |∂u/∂x|² + |∂u/∂y|² + |∂u/∂z|²。"""
    n = x.shape[0]
    eye = torch.eye(3, dtype=x.dtype) * d
    pts = torch.cat([x + eye[i] for i in range(3)], dim=0)
    u_all = model(pts)
    u0 = model(x)
    pen = 0.0
    for i in range(3):
        gi = (u_all[i * n:(i + 1) * n] - u0) / d
        pen = pen + (gi ** 2).sum(dim=1)
    return pen.mean()


def _train(model: nn.Module, Xtr, Ytr, Xte, Yte, epochs: int = EPOCHS,
           lr: float = LR, reg_lambda: float = 0.0, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = ((model(Xtr) - Ytr) ** 2).mean()
        if reg_lambda > 0:
            loss = loss + reg_lambda * _smoothness_penalty(model, Xtr, 2.0 / (SIZE - 1))
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            with torch.no_grad():
                te = float(((model(Xte) - Yte) ** 2).mean())
            hist.append({"epoch": ep, "训练 MSE": round(float(loss.item()), 4),
                         "测试 MSE": round(te, 4)})
    return {"耗时 (s)": round(time.time() - t0, 3), "history": hist}


def _endpoint_error(pred: np.ndarray, true: np.ndarray) -> float:
    """端点误差（mm）：每个采样点上预测位移与真值位移之差的模长均值。"""
    return float(np.linalg.norm(pred - true, axis=1).mean())


def _forward_grid(model: nn.Module, size: int, n: int) -> np.ndarray:
    """在 n³ 的均匀网格上求位移场，返回 (3, n, n, n)。"""
    g = np.linspace(0, size - 1, n)
    idx = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
    flat = _norm_coords(idx.reshape(-1, 3), size)
    with torch.no_grad():
        out = model(flat).numpy()
    return out.reshape(n, n, n, 3).transpose(3, 0, 1, 2)


def _predict_grid(model: nn.Module, size: int, coarse: int | None = None) -> np.ndarray:
    """在整幅网格上求位移场（可选：先在粗网格求值再线性插值上来）。"""
    if coarse is None:
        return _forward_grid(model, size, size)
    out = _forward_grid(model, size, coarse)
    zoom = (1.0, size / coarse, size / coarse, size / coarse)
    return ndimage.zoom(out, zoom, order=1)


def _timeit(fn, repeats: int = 7) -> float:
    """重复计时取最小值（最小值比均值更接近真实计算代价）。"""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _fmt_hist(hist: list[dict]) -> str:
    """把训练历史压成一行，便于写进 note。"""
    return " | ".join(f"{h['epoch']}ep:{h['测试 MSE']:.3f}" for h in hist)


# ----------------------------------------------------------------------
# META
# ----------------------------------------------------------------------
META = dict(
    id="R15",
    year=2025,
    title="KAN 隐式配准：用 Jacobi 多项式网络拟合形变场",
    journal="Bioengineering 2025;12(9):1005",
    doi="10.3390/bioengineering12091005",
    position="○ 合作者（第 3 / 5 作者）",
    slug="kan-implicit-registration",
    goal="把医学图像配准写成「坐标 → 位移」的连续函数拟合问题，手写 Jacobi 多项式 KAN 层，"
         "验证 KAN 能否在参数量与 MLP 相当时达到可比甚至更好的形变场拟合精度，"
         "并检验速度正则化能否消除 Jacobian 折叠。",
    difference="论文在真实 IGRT 图像上做端到端配准（含图像相似度项、时变速度场与真实 PCA 加速）；"
               "本复现把任务简化为「已知真值 DVF 的坐标回归」，只用 32³ 合成体模、"
               "1400 个采样点、300 个 epoch、CPU 训练，因此只比较 KAN 与 MLP 的"
               "相对精度/参数量/耗时，不比较绝对配准精度，也不复现原文的时变速度场公式。",
    conclusion=(
        "在合成形变场回归任务上，手写的 Jacobi 多项式 KAN（390 个参数）与参数量几乎相同的 "
        "Tanh-MLP（387 个参数）相比：固定 300 个 epoch 时 KAN 的测试端点误差为 0.110 mm，"
        "约为 MLP（0.300 mm）的 1/3；把训练预算提到 900 个 epoch，两者分别降到 0.054 mm 与 "
        "0.110 mm —— MLP 需要约 900 个 epoch 才能追上 KAN 在 300 个 epoch 时的精度，"
        "也就是 KAN 用约 1/3 的训练轮数达到了同等精度。但 KAN 每个 epoch 的耗时约为 MLP 的 "
        "2.5 倍（因为它要额外算多项式基），折算到墙钟时间两者大致相当，MLP 靠更多轮数可以追平。"
        "所以多项式基在这里带来的是「更少轮数、更少调参达到同等精度」的收敛效率，"
        "而不是纯粹的墙钟加速 —— 这一点比原文的表述更保守。"
        "隐式连续场表示还允许先在 16³ 粗网格上求值再线性插值到 32³：网络前向的计算量按点数"
        "降到 1/8、实测前向耗时降到约 1/9，但端到端（含插值上采样）总耗时只降到 1.1~1.3 倍，"
        "因为插值本身有固定开销；器官内体素级差异均值 0.037 mm、最大 0.16 mm。"
        "这恰好解释了原文为什么要进一步对速度场做主成分/低秩建模，而不是只靠降采样。"
        "物理合理性方面：干净目标下两个模型的形变场都没有出现 Jacobian 折叠；"
        "把采样点位移加上 1 mm 噪声、采样点减到 400 个之后，无正则模型的最差 det(J) 掉到 0.34"
        "（逼近折叠边缘）、端点误差 0.59 mm，而加入轻微速度正则化（λ=0.005）后端点误差降到 "
        "0.39 mm、最差 det(J) 回到 0.69 —— 速度正则化同时改善了精度与物理合理性，"
        "与原文「保证形变平滑、拓扑保持」的主张方向一致。需要强调：本复现只覆盖「形变场拟合」"
        "这一环，完整的配准还需要图像相似度驱动，不能据此评价原文端到端配准的精度。"),
    learn=[
        "KAN 层的本质：输出 = Σ_j φ_j(x_j)，把「激活函数」从固定非线性换成可学习的基函数展开",
        "Jacobi 多项式如何用三点递推实现，以及为什么要用 √h_n 归一化（正交归一基数值更稳）",
        "把配准写成隐式连续函数 u(x,y,z) 的收益：任意分辨率解析求值、天然支持粗网格+插值加速",
        "用 Jacobian 行列式判断形变是否物理合理：det(J) ≤ 0 意味着组织自我折叠",
        "速度/位移场正则化（一阶有限差分罚项）如何在几乎不损失精度的前提下压掉折叠",
    ],
    exercises=[
        "把 DEGREE 从 4 提到 8，观察 KAN 的拟合误差与过拟合（测试误差先降后升）的转折点",
        "把 N_TRAIN 从 1400 降到 300，比较 KAN 与 MLP 在小样本下的样本效率差异",
        "用 registration.register 跑 Demons 配准，比较「图像驱动配准」与「真值场回归」两条路线的误差量级",
    ],
)


# ----------------------------------------------------------------------
# 步骤
# ----------------------------------------------------------------------
def steps() -> list[Step]:
    def s1(ctx):
        """生成合成配准病例，并采样「坐标 → 位移」训练/测试集。"""
        case = make_registration_case(size=SIZE, magnitude=MAGNITUDE,
                                      n_landmarks=12, seed=0)
        rng = np.random.RandomState(0)
        idx = np.argwhere(case.organ)
        rng.shuffle(idx)

        n_use = min(N_TRAIN + N_TEST, len(idx))
        pts = idx[:n_use]
        coords = _norm_coords(pts, SIZE)
        target = case.dvf_true[:, pts[:, 0], pts[:, 1], pts[:, 2]].T   # (n, 3)
        target = np.ascontiguousarray(target.astype(np.float32))

        ctx.update(case=case,
                   Xtr=coords[:N_TRAIN], Ytr=torch.tensor(target[:N_TRAIN]),
                   Xte=coords[N_TRAIN:n_use], Yte=torch.tensor(target[N_TRAIN:n_use]),
                   pts_tr=pts[:N_TRAIN], pts_te=pts[N_TRAIN:n_use])

        disp = np.linalg.norm(target, axis=1)
        f_true = folding_stats(case.dvf_true, case.organ)
        return {
            "体模尺寸": f"{SIZE}³ 体素 @ 1 mm",
            "器官体素数": int(case.organ.sum()),
            "真值形变幅度 (mm)": f"±{MAGNITUDE:g}",
            "采样点平均位移 (mm)": round(float(disp.mean()), 3),
            "训练 / 测试采样点数": f"{len(ctx['Xtr'])} / {len(ctx['Xte'])}",
            "真值形变 负 Jacobian 比例": f_true["负 Jacobian 比例"],
            "真值形变 最差 det(J)": f_true["最差 det"],
            "真值形变判断": f_true["判断"],
        }

    def s2(ctx):
        """实现 Jacobi 多项式基并做数值验证（对照 scipy、检验正交归一性）。"""
        x = np.linspace(-1.0, 1.0, 11)
        mine = jacobi_basis_np(x, DEGREE, ALPHA, BETA)

        # (1) 与 scipy 的 Jacobi 多项式对照（先去掉归一化再比）
        raw = jacobi_basis_np(x, DEGREE, ALPHA, BETA)
        h = [2.0 ** (ALPHA + BETA + 1) / (2 * n + ALPHA + BETA + 1)
             * gamma(n + ALPHA + 1) * gamma(n + BETA + 1)
             / (gamma(n + ALPHA + BETA + 1) * gamma(n + 1))
             for n in range(DEGREE + 1)]
        ref = np.stack([eval_jacobi(n, ALPHA, BETA, x) for n in range(DEGREE + 1)], -1)
        dev_scipy = float(np.abs(raw * np.sqrt(h) - ref).max())

        # (2) torch 实现 vs numpy 参考实现
        tb = JacobiBasis(DEGREE, ALPHA, BETA)
        with torch.no_grad():
            dev_torch = float(np.abs(tb(torch.tensor(x, dtype=torch.float32)).numpy()
                                     - mine).max())

        # (3) 正交归一性：24 点 Gauss–Jacobi 求积下 ∫φ_i φ_j (1−x)^α(1+x)^β dx = δ_ij
        nodes, w = roots_jacobi(24, ALPHA, BETA)
        B = jacobi_basis_np(nodes, DEGREE, ALPHA, BETA)
        gram = (B * w[:, None]).T @ B
        off = float(np.abs(gram - np.eye(DEGREE + 1)).max())

        kan = JacobiKAN()
        mlp = make_mlp()
        ctx["basis_table"] = pd.DataFrame({
            "阶数 n": list(range(DEGREE + 1)),
            "φ_n(−1)": np.round(mine[0], 4),
            "φ_n(0)": np.round(mine[len(x) // 2], 4),
            "φ_n(+1)": np.round(mine[-1], 4),
            "对角元 ∫φ²": np.round(np.diag(gram), 6),
        })
        ctx["param_counts"] = {"KAN": n_params(kan), "MLP": n_params(mlp)}
        return {
            "基函数族": f"归一化 Jacobi P_n^({ALPHA:g},{BETA:g})，n = 0..{DEGREE}",
            "与 scipy.eval_jacobi 最大偏差": f"{dev_scipy:.2e}",
            "torch 实现与 numpy 参考最大偏差": f"{dev_torch:.2e}",
            "Gram 矩阵最大非对角元（应≈0）": f"{off:.2e}",
            "KAN 参数量": n_params(kan),
            "MLP 参数量": n_params(mlp),
        }

    def s3(ctx):
        """基线：坐标 MLP 拟合真值 DVF。"""
        mlp = make_mlp()
        info = _train(mlp, ctx["Xtr"], ctx["Ytr"], ctx["Xte"], ctx["Yte"], seed=0)
        with torch.no_grad():
            pred_te = mlp(ctx["Xte"]).numpy()
        ctx["mlp"] = mlp
        ctx["hist_mlp"] = info["history"]
        ctx["res_mlp"] = {
            "模型": "坐标 MLP（Tanh）",
            "参数量": n_params(mlp),
            "训练轮数": EPOCHS,
            "训练耗时 (s)": info["耗时 (s)"],
            "测试端点误差 (mm)": round(_endpoint_error(pred_te, ctx["Yte"].numpy()), 4),
            "测试 MSE": info["history"][-1]["测试 MSE"],
            "训练历史（测试 MSE@epoch）": _fmt_hist(info["history"]),
        }
        return ctx["res_mlp"]

    def s4(ctx):
        """KAN 拟合 DVF：两层 Jacobi-KAN，与 MLP 同样的训练预算。"""
        kan = JacobiKAN()
        info = _train(kan, ctx["Xtr"], ctx["Ytr"], ctx["Xte"], ctx["Yte"], seed=0)
        with torch.no_grad():
            pred_te = kan(ctx["Xte"]).numpy()
        ctx["kan"] = kan
        ctx["hist_kan"] = info["history"]
        ctx["res_kan"] = {
            "模型": "Jacobi-KAN",
            "参数量": n_params(kan),
            "训练轮数": EPOCHS,
            "训练耗时 (s)": info["耗时 (s)"],
            "测试端点误差 (mm)": round(_endpoint_error(pred_te, ctx["Yte"].numpy()), 4),
            "测试 MSE": info["history"][-1]["测试 MSE"],
            "训练历史（测试 MSE@epoch）": _fmt_hist(info["history"]),
        }
        ctx["param_counts"]["KAN"] = n_params(kan)
        return ctx["res_kan"]

    def s5(ctx):
        """精度—参数量—训练预算对比：同一精度下谁更省。"""
        rows = [dict(ctx["res_mlp"]), dict(ctx["res_kan"])]

        # 把训练预算提到 1200 epoch，看两种基函数在充分训练下的差距
        for name, model in [("坐标 MLP（Tanh）", make_mlp()), ("Jacobi-KAN", JacobiKAN())]:
            info = _train(model, ctx["Xtr"], ctx["Ytr"], ctx["Xte"], ctx["Yte"],
                          epochs=EPOCHS_LONG, seed=0)
            with torch.no_grad():
                p = model(ctx["Xte"]).numpy()
            rows.append({
                "模型": name, "参数量": n_params(model), "训练轮数": EPOCHS_LONG,
                "训练耗时 (s)": info["耗时 (s)"],
                "测试端点误差 (mm)": round(_endpoint_error(p, ctx["Yte"].numpy()), 4),
                "测试 MSE": info["history"][-1]["测试 MSE"],
            })
        df = pd.DataFrame(rows)
        best = float(df["测试端点误差 (mm)"].min())
        df["相对最佳误差"] = np.round(df["测试端点误差 (mm)"] / best, 2)
        df["每 epoch 耗时 (ms)"] = np.round(df["训练耗时 (s)"] / df["训练轮数"] * 1000, 3)
        ctx["cmp_table"] = df
        return df

    def s6(ctx):
        """隐式连续场的加速验证：粗网格求值 + 插值 vs 全网格求值。"""
        case, kan = ctx["case"], ctx["kan"]
        # 网络前向本身的计算量：32³ = 32768 点 vs 16³ = 4096 点（相差 8 倍）
        t_fwd_full = _timeit(lambda: _forward_grid(kan, SIZE, SIZE))
        t_fwd_coarse = _timeit(lambda: _forward_grid(kan, SIZE, COARSE))
        # 端到端（含把粗网格线性插值上采样到 32³ 的开销）
        t_full = _timeit(lambda: _predict_grid(kan, SIZE))
        t_coarse = _timeit(lambda: _predict_grid(kan, SIZE, coarse=COARSE))

        grid_full = _predict_grid(kan, SIZE)
        grid_coarse = _predict_grid(kan, SIZE, coarse=COARSE)
        ctx["grid_kan"] = grid_full
        diff = np.linalg.norm(grid_full - grid_coarse, axis=0)[case.organ]
        return {
            f"网络前向 {SIZE}³ / {SIZE ** 3} 点 (s)": round(t_fwd_full, 4),
            f"网络前向 {COARSE}³ / {COARSE ** 3} 点 (s)": round(t_fwd_coarse, 4),
            "网络前向加速比": round(t_fwd_full / max(t_fwd_coarse, 1e-9), 2),
            "端到端（含插值）全网格 (s)": round(t_full, 4),
            "端到端（含插值）粗网格 (s)": round(t_coarse, 4),
            "端到端加速比": round(t_full / max(t_coarse, 1e-9), 2),
            "耗时下降（端到端）": f"{(1 - t_coarse / max(t_full, 1e-9)) * 100:.1f}%",
            "两种求值的体素级差异 均值 (mm)": round(float(diff.mean()), 4),
            "两种求值的体素级差异 最大值 (mm)": round(float(diff.max()), 4),
            "说明": "计算量按点数降到 1/8；端到端受插值上采样开销限制，"
                    "这正是原文要用主成分建模速度场来进一步压缩的原因",
        }

    def s7(ctx):
        """Jacobian 折叠检测：干净目标 vs 带噪目标 + 速度正则化扫描。"""
        case = ctx["case"]
        rows = []

        def stats(dvf, err=None):
            f = folding_stats(dvf, case.organ)
            g = np.gradient(dvf)
            gmag = round(float(np.sqrt(sum((np.asarray(gi) ** 2).sum(axis=0)
                                          for gi in g)).mean()), 4)
            return {"测试端点误差 (mm)": (None if err is None else round(err, 4)),
                    "负 Jacobian 比例": f["负 Jacobian 比例"],
                    "折叠体素数": f["折叠体素数"],
                    "最差 det(J)": f["最差 det"],
                    "det(J) 标准差": f["det 标准差"],
                    "平均梯度幅值": gmag,
                    "判断": f["判断"]}

        # (1) 真值形变（上界参照）
        rows.append({"实验设置": "真值形变（参照）", "形变场来源": "解析真值",
                     **stats(case.dvf_true)})
        # (2) 干净目标下的两个模型
        for name, model, res in [("干净目标（300ep）", ctx["mlp"], ctx["res_mlp"]),
                                 ("干净目标（300ep）", ctx["kan"], ctx["res_kan"])]:
            g = _predict_grid(model, SIZE)
            rows.append({"实验设置": name, "形变场来源": res["模型"],
                         **stats(g, res["测试端点误差 (mm)"])})

        # (3) 带噪目标：模拟标志点定位 / 图像噪声，扫描速度正则化权重
        n = N_NOISY
        rng = np.random.RandomState(7)
        Y_noisy = ctx["Ytr"][:n] + torch.tensor(
            rng.randn(n, 3).astype(np.float32) * NOISE_SIGMA)
        for lam in REG_LAMBDAS:
            m = JacobiKAN()
            _train(m, ctx["Xtr"][:n], Y_noisy, ctx["Xte"], ctx["Yte"],
                   reg_lambda=lam, seed=0)
            g = _predict_grid(m, SIZE)
            with torch.no_grad():
                p = m(ctx["Xte"]).numpy()
            err = _endpoint_error(p, ctx["Yte"].numpy())
            lab = (f"带噪目标 σ={NOISE_SIGMA:g}mm / {n} 点"
                   + (f" + 速度正则 λ={lam:g}" if lam > 0 else "（无正则）"))
            rows.append({"实验设置": lab, "形变场来源": "Jacobi-KAN",
                         **stats(g, err)})

        # (4) 传统迭代配准作为参照（图像驱动，任务不同）
        reg = register(case, "fast_symmetric", iterations=60)
        ev = evaluate_registration(reg, case)
        rows.append({"实验设置": "参照：图像驱动配准", "形变场来源": "Demons",
                     "测试端点误差 (mm)": None,
                     "负 Jacobian 比例": ev["负 Jacobian 比例"],
                     "折叠体素数": ev["折叠体素数"],
                     "最差 det(J)": ev["最差 det"],
                     "det(J) 标准差": ev["det 标准差"],
                     "平均梯度幅值": None,
                     "判断": f"DVF 端点误差 {ev['DVF 端点误差 (mm)']} mm，"
                             f"TRE {ev['TRE (mm)']} mm，耗时 {ev['耗时 (s)']} s"})

        return pd.DataFrame(rows)

    return [
        Step("① 生成合成配准病例并采样坐标-位移训练集",
             "对应原文的数据准备环节。用已知真值形变场的合成体模（32³、幅度 ±4 mm），"
             "在器官内随机采样 2000 个体素，取「归一化坐标 → 真值位移」作为回归样本，"
             "前 1400 个训练、后 600 个测试。",
             s1, "metrics",
             "真值形变本身是平滑的（高斯随机场），负 Jacobian 比例为 0，可作为物理合理性的上界。"),
        Step("② 手写 Jacobi 多项式基并做数值验证",
             "对应原文用 Jacobi 多项式构造 KAN 的关键实现。用三点递推实现归一化 Jacobi 基，"
             "并与 scipy.eval_jacobi 对照；用 24 点 Gauss–Jacobi 求积检验正交归一性；"
             "再确认 torch 版本与 numpy 参考实现完全一致。",
             s2, "metrics",
             "正交归一基让各阶系数互不干扰，也让不同阶数的模型可以直接比较系数幅值。"),
        Step("③ 基线：坐标 MLP 拟合 DVF",
             "对应原文的对照组。用 Tanh 全连接网络（3-16-16-3，387 个参数）拟合同样的"
             "「坐标 → 位移」映射，训练 300 个 epoch，记录耗时、测试 MSE 与端点误差。",
             s3, "metrics",
             "MLP 的参数量刻意调到与 KAN（390）几乎相同，这样比较的是「基函数形式」而非「模型容量」。"),
        Step("④ Jacobi-KAN 拟合 DVF",
             "对应原文的核心方法。用两层 KAN（3→13→3，每层 13×3×5 与 3×13×5 个可学习系数，"
             "层间用 tanh 把隐层激活压回基函数的定义域）在与 MLP 完全相同的训练预算下拟合 DVF。",
             s4, "metrics",
             "KAN 层输出 = Σ_j Σ_k c_{o,j,k} φ_k(x_j)，φ_k 就是用第②步验证过的 Jacobi 基。"),
        Step("⑤ 精度—参数量—训练预算对比",
             "对应原文报告的效率优势。除了 300 epoch 的等预算对照，额外把两种模型都训练到 "
             "900 epoch，观察「充分训练后差距是否消失」，并给出每 epoch 耗时，"
             "以便把「收敛效率」与「单步算力」分开看。",
             s5, "table",
             "KAN 每 epoch 更贵但收敛更快；这张表回答的是「同一个精度谁更省」而不是「谁更准」。"),
        Step("⑥ 隐式连续场的加速验证（粗网格求值 + 插值）",
             "对应原文用主成分/低秩建模速度场来提速的思路：KAN 是坐标的连续函数，"
             "可以先在 16³ 粗网格上求值，再线性插值到 32³，用很小的精度代价换前向耗时下降。",
             s6, "metrics",
             "这一步量化的是「连续场表示」本身的加速潜力，不是原文 PCA 加速的等价实现。"),
        Step("⑦ Jacobian 折叠检测：干净目标 vs 带噪目标 + 速度正则化",
             "对应原文的速度正则化质控。把各模型在整幅 32³ 网格上求值并计算 Jacobian 行列式："
             "det(J) ≤ 0 代表组织自折叠。除了干净目标下的两个模型，还把采样点位移加上 "
             "1 mm 噪声、采样点减到 400 个（模拟标志点定位误差与稀疏采样），"
             "扫描速度正则化权重 λ，看它能否同时改善精度与物理合理性。",
             s7, "table",
             "正则化罚的是 |∂u/∂x|²：λ 越大最差 det(J) 越接近 1，但过大又会把真实形变一起抹平。"),
    ]
