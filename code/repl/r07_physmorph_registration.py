"""
R07 · PhysMorph：形变配准与物理合理性
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r7_steps

META = dict(
    id="R07",
    year=2026,
    title="PhysMorph：形变配准与物理合理性",
    journal="Physics and Imaging in Radiation Oncology 2026;37:100906",
    doi="10.1016/j.phro.2026.100906",
    position="☆ 通讯/末位",
    slug="physmorph-registration",
    goal="""把有限元力学约束嵌入深度学习配准；用 Jacobian 行列式证明形变物理合理，并把 10 分钟的传统方法加速到 103 毫秒。""",
    difference="""论文用 42 例肝 SBRT（MRI-CBCT 与 MR-Linac）；平台用已知形变的合成体模 + SimpleITK 的 Demons 算法，复现**同一套评估体系**（TRE / MSD / Jacobian）。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "形变配准的评估体系（TRE / MSD）",
        "Jacobian 行列式如何判断形变是否物理合理",
        "传统迭代配准的精度与代价",
    ],
    exercises=[
        "把形变幅度从 3.5 增到 7，看 TRE 与折叠比例的变化",
        "调整迭代次数，观察精度与耗时的权衡",
        "对配准结果做高斯平滑，看折叠是否减少",
    ],
)


def steps() -> list[Step]:
    return _r7_steps()
