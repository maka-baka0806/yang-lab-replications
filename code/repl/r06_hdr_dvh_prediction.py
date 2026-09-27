"""
R06 · HDR 近距离治疗：个性化 DVH 预测
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r6_steps

META = dict(
    id="R06",
    year=2022,
    title="HDR 近距离治疗：个性化 DVH 预测",
    journal="Frontiers in Oncology 2022;12:967436",
    doi="10.3389/fonc.2022.967436",
    position="○ 第三作者",
    slug="hdr-dvh-prediction",
    goal="""PCA 提特征 → kNN 找相似历史病例 → KDE 建立「距离-剂量」概率模型，预测膀胱/直肠的 D2cc 等剂量指标。""",
    difference="""论文用 79 例宫颈癌真实 Oncentra 计划；平台用参数化合成病例模拟距离-剂量关系，复现**PCA+kNN+KDE 的完整流程**。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "DVH 与 D2cc/Dmean 等剂量指标的物理含义",
        "PCA + kNN 的相似病例检索流程",
        "核密度估计为什么比单点预测更有价值",
    ],
    exercises=[
        "把 k 从 20 改到 5 或 40，看预测残差如何变化",
        "改变距离-剂量关系的随机参数范围，观察泛化性",
        "把 KDE 换成直接用 kNN 均值，比较残差",
    ],
)


def steps() -> list[Step]:
    return _r6_steps()
