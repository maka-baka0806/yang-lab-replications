"""
R08 · Neural ODE：把网络行为画成轨迹
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r8_steps

META = dict(
    id="R08",
    year=2023,
    title="Neural ODE：把网络行为画成轨迹",
    journal="Med Phys 2023;50(8):4825-4838 / 2025;52(4):2661-2674 / IJROBP 2026;125(2):649-659",
    doi="10.1002/mp.16286",
    position="★ 第一作者（首篇）/ ○ 合作者（后两篇）",
    slug="neural-ode-trajectory",
    goal="""把网络层间传递建模为连续时间演化 dz/dt = f(z,t)，于是可以画出每个样本「走向结论的轨迹」——把黑箱变成可观察的动力系统。""",
    difference="""论文处理 MP-MRI 与影像基因组学数据并需 GPU 训练；平台在二维合成数据上手写 RK4 积分器，复现**同一机制**与轨迹可视化。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "神经常微分方程与 RK4 积分器的实现",
        "如何把网络行为可视化为潜空间轨迹",
        "为什么参数量小是 Neural ODE 的优势",
    ],
    exercises=[
        "把积分步数从 8 增到 16，看精度与耗时的变化",
        "换用 spiral 数据集（更难），观察轨迹分离度",
        "把重球阻尼系数改大或改小，比较收敛行为",
    ],
)


def steps() -> list[Step]:
    return _r8_steps()
