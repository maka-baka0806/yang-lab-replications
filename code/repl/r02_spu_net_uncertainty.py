"""
R02 · SPU-Net：让分割模型说出「我有多确定」
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r2_steps

META = dict(
    id="R02",
    year=2024,
    title="SPU-Net：让分割模型说出「我有多确定」",
    journal="Medical Physics 2024;51(3):1931-1943 / 2026;53(3):e70360",
    doi="10.1002/mp.16695",
    position="★ 第一作者（含国自然青年项目）",
    slug="spu-net-uncertainty",
    goal="""用球面投影制造多视角 → 多组预测 → **方差即不确定性**；再用 Otsu 聚合提升分割精度，并验证不确定性可预测错误。""",
    difference="""论文用 369 例 BraTS 胶质瘤 MP-MRI 训练 U-Net；平台用合成病灶体模 + 经典分割算法复现**同一套不确定性机制**，无需 GPU 训练。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "SPU-Net 的核心机制：多视角预测，方差即不确定性",
        "如何用 Otsu 聚合多组预测得到共识分割",
        "如何验证不确定性确实能预测错误",
    ],
    exercises=[
        "把扰动视角数从 12 增到 20，共识 Dice 是否继续提升",
        "把旋转角上限从 12 度改到 30 度，观察不确定性的变化",
        "把基础分割算法从 Otsu 换成 watershed，看结论是否稳健",
    ],
)


def steps() -> list[Step]:
    return _r2_steps()
