"""
R09 · 4DCT 体素级时序放射组学：让 CT 变成「呼吸电影」
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r9_steps

META = dict(
    id="R09",
    year=2025,
    title="4DCT 体素级时序放射组学：让 CT 变成「呼吸电影」",
    journal="arXiv:2503.23898（2025）",
    doi="",
    position="☆ 通讯/末位",
    slug="4dct-radiomic-sequence",
    goal="""把静态 CT 升级为呼吸周期时间序列（体素级特征序列），用带时间显著性的时序模型判别通气缺陷；结论：**呼气相时受损区强度上升、均匀性下降**。""",
    difference="""论文用 45 例 VAMPIRE 真实 4DCT + 56 维特征 + LSTM；平台用合成 4DCT + 2 个特征的统计量 + 逻辑回归，复现**生理机制与结论方向**，绝对性能不可比。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "4DCT 的呼吸相位与通气缺损的影像表现",
        "体素级特征时间序列的构造方法",
        "时间显著性分析：哪些呼吸相位最具判别力",
    ],
    exercises=[
        "把相位数从 10 减到 5，看 AUC 下降多少",
        "改变缺损半径，观察强度斜率的变化",
        "把滑窗核从 5 改到 9，看均匀性序列的平滑程度",
    ],
)


def steps() -> list[Step]:
    return _r9_steps()
