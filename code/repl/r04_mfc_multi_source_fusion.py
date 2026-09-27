"""
R04 · MFC：手工特征 + 深度特征 + 临床信息三源融合
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r4_steps

META = dict(
    id="R04",
    year=2023,
    title="MFC：手工特征 + 深度特征 + 临床信息三源融合",
    journal="Frontiers in Oncology 2023;13:1185771",
    doi="10.3389/fonc.2023.1185771",
    position="★ 第一作者",
    slug="mfc-multi-source-fusion",
    goal="""三分支融合（105 手工特征 + 512 深度特征 + 4 临床信息）预测早期 NSCLC 局部失败，AUC 显著高于任何单一来源。""",
    difference="""论文用真实手术/SBRT 队列；平台用合成队列（按「单源信息不完整」原则生成），复现**融合优于单源**这一核心结论，绝对 AUC 不可直接比较。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "多共线性评估（VIF）与冗余特征剔除",
        "三种交叉验证方案（k 折 / 留一 / 蒙特卡洛）的差别",
        "如何用置换重要性解释模型",
    ],
    exercises=[
        "把信号强度调低，看融合优势是否仍然存在",
        "把 LOOCV 换成 MCCV，比较 AUC 的稳定性",
        "删除临床特征，看融合模型的 AUC 下降多少",
    ],
)


def steps() -> list[Step]:
    return _r4_steps()
