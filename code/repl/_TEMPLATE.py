"""
复现模块模板
==============
复制本文件为 rXX_<slug>.py 并填写。契约：

    META: dict   必须包含 id / year / title / journal / doi / position /
                 goal / difference / slug / conclusion / learn / exercises
    steps() -> list[Step]

每个 Step: Step(title, detail, fn, kind, note)
    - fn(ctx) 接收共享状态字典 ctx，可把中间结果存进 ctx 供后续步骤使用
    - kind ∈ {"text", "metrics", "table"}
    - 步骤必须是**可独立复现**的：不依赖随机性之外的外部输入
"""
from __future__ import annotations

from framework import Step

META = dict(
    id="R00",
    year=2025,
    title="示例：把论文方法拆成可执行步骤",
    journal="Journal Name 2025;1(1):1-10",
    doi="10.0000/example",
    position="★ 第一作者",
    slug="template",
    goal="一句话说明这篇论文做了什么、复现要验证什么。",
    difference="论文用真实临床数据；本复现用合成数据，因此只比较方法学方向。",
    conclusion="复现成功后写下结论，会出现在报告末尾。",
    learn=["你会学到的第一件事", "第二件事"],
    exercises=["延伸练习一", "延伸练习二"],
)


def steps() -> list[Step]:
    def s1(ctx):
        ctx["value"] = 42
        return {"示例指标": 42, "说明": "第一步的产出会存入 ctx"}

    def s2(ctx):
        return {"上一步的值": ctx["value"], "乘二": ctx["value"] * 2}

    return [
        Step("① 第一步", "描述这一步在做什么、对应原文哪个环节。",
             s1, "metrics", "可选：这一步的结论"),
        Step("② 第二步", "演示如何在步骤间传递状态。", s2, "metrics"),
    ]
