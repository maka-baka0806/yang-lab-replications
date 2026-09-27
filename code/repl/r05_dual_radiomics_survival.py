"""
R05 · 双放射组学 + 生存分层
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r5_steps

META = dict(
    id="R05",
    year=2024,
    title="双放射组学 + 生存分层",
    journal="Frontiers in Oncology 2024;14:1419621",
    doi="10.3389/fonc.2024.1419621",
    position="☆ 通讯/末位（带学生成果）",
    slug="dual-radiomics-survival",
    goal="""手工放射组学 + 深度放射组学双路建模预测总生存；风险分数中位分层后用 Kaplan-Meier 与 log-rank 检验比较两组生存。""",
    difference="""论文用 TCIA Lung1 的 132 例真实数据；平台用合成生存数据复现**下游分析链**（分层 → KM → log-rank），KM 与 log-rank 为手工实现。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "Kaplan-Meier 估计的手工实现（含 Greenwood 方差）",
        "log-rank 检验的原理与实现",
        "风险分数分层（中位数 vs 三分位）",
    ],
    exercises=[
        "把删失比例从 25% 提到 50%，看 log-rank 的 p 值变化",
        "把风险分数加噪声，观察生存曲线分离度下降",
        "把中位分层改成三分位，比较两组的统计功效",
    ],
)


def steps() -> list[Step]:
    return _r5_steps()
