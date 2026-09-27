"""
R03 · SCNN：把大脑装进球里预测剂量
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r3_steps

META = dict(
    id="R03",
    year=2025,
    title="SCNN：把大脑装进球里预测剂量",
    journal="Medical Physics 2025;52(6):4266-4277",
    doi="10.1002/mp.17748",
    position="★ 第一作者",
    slug="scnn-spherical-projection",
    goal="""球形投影把 3D 靶区分布压成 2D 球面图，用球面卷积网络预测正常脑 V50%/V60%/V66.7%，参数量仅为 3D U-Net 的 1/30。""",
    difference="""论文用 106 例真实 SIMT 计划（Eclipse + AAA）训练 SCNN，R² 达 0.92–0.94；平台复现核心几何变换与「球面图编码剂量信息」的前提，但不训练网络。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "球形投影：把 3D 几何压成 2D 球面图的核心变换",
        "SIMT 剂量学指标（V50%/V60%/V66.7%、V10Gy/V12Gy）",
        "为什么球面表示能大幅降低参数量",
    ],
    exercises=[
        "把靶点数从 4 改到 8，观察球面图与剂量指标的关系",
        "改处方剂量，看 V50% 与 V12Gy 的对应关系如何变化",
        "用球面熵做单变量预测，与多特征回归比较",
    ],
)


def steps() -> list[Step]:
    return _r3_steps()
