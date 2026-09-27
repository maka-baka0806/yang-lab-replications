"""
神经常微分方程与决策轨迹可视化（Neural ODE）
==============================================
对应文献：
  - Yang Z, et al. A neural ordinary differential equation model for visualizing deep
    neural network behaviors in multi-parametric MRI-based glioma segmentation.
    Med Phys 2023;50(8):4825-4838.
  - Zhao J, Vaios E, Yang Z, et al. Radiogenomic explainable AI with neural ordinary
    differential equation … Med Phys 2025;52(4):2661-2674.（HBNODE）
  - Zhao J, …, Yang Z, et al. An Explainable Deep Model for Risk Scoring … IJROBP
    2026;125(2):649-659.（LRP + 决策场 F）

论文的核心思想：把神经网络的层间传递看成**连续时间演化**
    dz/dt = f(z, t)
于是可以把每个样本「走向结论的轨迹」画出来 —— 这是把黑箱变成可观察动力系统的一条路径。

本模块用 PyTorch 手写 RK4 积分器（不依赖 torchdiffeq），在二维合成数据上
训练一个 ODE-Net，并导出轨迹用于可视化。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NODEResult:
    trajectories: np.ndarray      # (n_times, n_samples, 2) 潜空间轨迹
    times: np.ndarray
    loss_history: list[float]
    accuracy: float
    n_params: int
    separation: np.ndarray = field(default_factory=lambda: np.zeros(0))


def make_spiral(n_per_class: int = 150, noise: float = 0.12,
                seed: int = 0, kind: str = "circles") -> tuple[np.ndarray, np.ndarray]:
    """生成二维二分类数据。

    kind：
      - "circles"（默认）：同心圆 —— 线性不可分但可稳定学习，适合演示轨迹
      - "spiral"：双螺旋 —— 更难，训练波动大
    """
    rng = np.random.RandomState(seed)
    n = n_per_class

    if kind == "circles":
        theta = rng.uniform(0, 2 * np.pi, n)
        r_in = rng.uniform(0.15, 0.45, n)
        r_out = rng.uniform(0.70, 1.00, n)
        x1 = np.column_stack([r_in * np.cos(theta), r_in * np.sin(theta)])
        theta2 = rng.uniform(0, 2 * np.pi, n)
        x2 = np.column_stack([r_out * np.cos(theta2), r_out * np.sin(theta2)])
    else:
        theta = np.linspace(0.4, 3.0 * np.pi, n)
        r = np.linspace(0.25, 1.0, n)
        x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        x2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])

    X = np.vstack([x1, x2]) + rng.randn(2 * n, 2) * noise
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    return X.astype(np.float32), y


def train_neural_ode(X: np.ndarray, y: np.ndarray, steps: int = 400,
                     hidden: int = 48, lr: float = 0.01,
                     n_times: int = 8, seed: int = 0) -> NODEResult:
    """训练一个 RK4 积分的 ODE-Net，并导出潜空间轨迹。

    dz/dt = MLP(z, t)，从 t=0 积分到 t=1，末态送线性分类头。
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    class ODEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 2),
            )

        def forward(self, t, z):
            tt = torch.full((z.shape[0], 1), float(t))
            return self.net(torch.cat([z, tt], dim=1))

    class ODENet(nn.Module):
        def __init__(self):
            super().__init__()
            self.func = ODEFunc()
            self.head = nn.Linear(2, 2)

        def rk4(self, z, t0, t1, n_steps=8):
            h = (t1 - t0) / n_steps
            t = t0
            for _ in range(n_steps):
                k1 = self.func(t, z)
                k2 = self.func(t + h / 2, z + h * k1 / 2)
                k3 = self.func(t + h / 2, z + h * k2 / 2)
                k4 = self.func(t + h, z + h * k3)
                z = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t = t + h
            return z

        def forward(self, z0, record_times=None):
            """返回末态 logits；若给定 record_times，同时返回轨迹。"""
            if record_times is None:
                z = self.rk4(z0, 0.0, 1.0)
                return self.head(z)
            traj, z, t_prev = [], z0, 0.0
            for t in record_times:
                if t > t_prev:
                    z = self.rk4(z, t_prev, float(t))
                    t_prev = float(t)
                traj.append(z)
            return self.head(z), torch.stack(traj, dim=0)

    model = ODENet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    losses = []
    for i in range(steps):
        opt.zero_grad()
        logits = model(Xt)
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
        if i % 10 == 0:
            losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        record = np.linspace(0, 1, n_times)
        logits, traj = model(Xt, record_times=record)
        acc = float((logits.argmax(1) == yt).float().mean())
        # 每个时间点的类间分离度（质心距离）
        sep = []
        for k in range(traj.shape[0]):
            z = traj[k].numpy()
            sep.append(float(np.linalg.norm(z[y == 0].mean(0) - z[y == 1].mean(0))))

    n_params = sum(p.numel() for p in model.parameters())
    return NODEResult(trajectories=traj.numpy(), times=record,
                      loss_history=losses, accuracy=acc, n_params=n_params,
                      separation=np.array(sep))
