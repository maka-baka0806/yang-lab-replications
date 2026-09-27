"""
R01 · 体素级放射组学滤波：从 CT 量化肺功能
======================================================================
从平台复现专栏迁移到独立项目；步骤定义见 code/common/paper_steps.py。
"""
from __future__ import annotations

from framework import Step
from common.paper_steps import _r1_steps

META = dict(
    id="R01",
    year=2022,
    title="体素级放射组学滤波：从 CT 量化肺功能",
    journal="Medical Physics 2022;49(11):7278-7286",
    doi="10.1002/mp.15837",
    position="★ 第一作者",
    slug="pulmonary-radiomic-filtering",
    goal="""把整个肺的放射组学分析扩展到**体素级**，得到空间分辨的特征图，并与 PET/SPECT 通气图做体素级相关，验证纹理特征携带通气信息。""",
    difference="""论文用 46 例 VAMPIRE 真实临床数据（含 Galligas PET / DTPA SPECT）；平台用合成肺体模（已知缺损区），因此只能比较**方法学方向**，不能比较绝对数值。""",
    conclusion="复现结论见执行后生成的报告。",
    learn=[
        "体素级放射组学滤波的完整实现（滑窗 + 多方向旋转不变）",
        "为什么纹理特征优于纯强度特征",
        "如何用核大小与分箱敏感性判断特征稳健性",
    ],
    exercises=[
        "把滑窗核从 15 mm 改到 25 mm，看相关性如何变化",
        "换一个缺损位置（右肺/中央），结论是否仍成立",
        "把 13 方向平均改成 3 方向，观察旋转不变性的影响",
    ],
)


def steps() -> list[Step]:
    return _r1_steps()
