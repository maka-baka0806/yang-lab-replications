"""
复现框架：统一的模块契约 + 运行器 + 报告生成器
================================================
每个复现模块（code/repl/rXX_*.py）只需暴露两个东西：

    META = dict(id="R01", year=2020, title=..., ...)
    def steps() -> list[Step]

运行器负责：按顺序执行、捕获结果与异常、记录耗时、生成 Markdown 报告。

用法：
    python code/run.py --list               列出全部复现
    python code/run.py --all                运行全部并生成报告
    python code/run.py --id R01             运行单篇
    python code/run.py --id R01 --stdout    只打印到终端，不写文件
"""
from __future__ import annotations

import importlib
import io
import pkgutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
PAPERS_DIR = ROOT / "papers"


# ----------------------------------------------------------------------
# 契约
# ----------------------------------------------------------------------
@dataclass
class Step:
    """一个可独立执行的复现步骤。"""
    title: str
    detail: str                       # 这一步做什么、对应原文哪个环节
    fn: Callable[[dict], Any]
    kind: str = "text"                # text | metrics | table | figure | code
    note: str = ""                    # 结论 / 与原文的差异提示


@dataclass
class StepResult:
    index: int
    step: Step
    ok: bool
    payload: Any = None
    error: str = ""
    seconds: float = 0.0


@dataclass
class RunResult:
    meta: dict
    steps: list[StepResult] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.steps if s.ok)

    @property
    def failed(self) -> list[StepResult]:
        return [s for s in self.steps if not s.ok]


# ----------------------------------------------------------------------
# 发现与执行
# ----------------------------------------------------------------------
def discover() -> list[tuple[str, Any]]:
    """发现全部复现模块，按 id 排序。"""
    import repl
    found = []
    for m in pkgutil.iter_modules(repl.__path__):
        if not m.name.startswith("r"):
            continue
        mod = importlib.import_module(f"repl.{m.name}")
        if hasattr(mod, "META") and hasattr(mod, "steps"):
            found.append((mod.META["id"], mod))
    return sorted(found, key=lambda x: x[0])


def run_module(mod: Any, progress: Callable[[str], None] | None = None) -> RunResult:
    """执行一个模块的全部步骤，捕获每步结果与异常。"""
    meta = dict(mod.META)
    res = RunResult(meta=meta)
    ctx: dict = {}
    t_all = time.time()

    for i, step in enumerate(mod.steps(), 1):
        if progress:
            progress(f"    [{i}] {step.title}")
        t0 = time.time()
        try:
            payload = step.fn(ctx)
            res.steps.append(StepResult(i, step, True, payload=payload,
                                        seconds=time.time() - t0))
        except Exception as e:
            res.steps.append(StepResult(
                i, step, False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
                seconds=time.time() - t0))
    res.total_seconds = time.time() - t_all
    return res


# ----------------------------------------------------------------------
# 报告生成
# ----------------------------------------------------------------------
def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)



def _df_to_markdown(df) -> str:
    """把 DataFrame 渲染成 Markdown 表格（不依赖 tabulate）。"""
    try:
        cols = [str(c) for c in df.columns]
        rows = []
        for _, r in df.iterrows():
            cells = []
            for v in r:
                if isinstance(v, float):
                    s = f"{v:.4g}"
                else:
                    s = str(v)
                cells.append(s.replace("|", "\\|").replace("\n", " "))
            rows.append(cells)
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in rows])
    except Exception:
        return "```\n" + str(df)[:2000] + "\n```"


def _render_payload(payload: Any, kind: str) -> str:
    """把步骤结果渲染成 Markdown 片段。"""
    if payload is None:
        return ""
    if kind == "metrics" and isinstance(payload, dict):
        rows = "\n".join(f"| {k} | {_fmt_value(v)} |" for k, v in payload.items())
        return "| 指标 | 数值 |\n|---|---|\n" + rows
    if kind == "table":
        return _df_to_markdown(payload)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        rows = "\n".join(f"| {k} | {_fmt_value(v)} |" for k, v in payload.items())
        return "| 项目 | 数值 |\n|---|---|\n" + rows
    return _df_to_markdown(payload)


def build_report(res: RunResult) -> str:
    m = res.meta
    L: list[str] = []
    L.append(f"# {m['id']} · {m['title']}")
    L.append("")
    L.append(f"> **原文**：{m['journal']}"
             + (f"　·　[{m['doi']}](https://doi.org/{m['doi']})" if m.get("doi") else "")
             + f"　·　{m['position']}")
    L.append(">")
    L.append(f"> **复现目标**：{m['goal']}")
    L.append(">")
    L.append(f"> **与原论文的差异**：{m['difference']}")
    L.append("")
    L.append(f"*报告生成时间：{datetime.now():%Y-%m-%d %H:%M}　·　"
             f"步骤 {res.ok_count}/{len(res.steps)} 通过　·　"
             f"总耗时 {res.total_seconds:.1f} 秒*")
    L.append("")
    L.append("---")
    L.append("")

    # 步骤导航
    L.append("## 步骤导航")
    L.append("")
    for s in res.steps:
        mark = "已完成" if s.ok else "**失败**"
        L.append(f"{s.index}. {s.step.title} —— {mark}（{s.seconds:.2f}s）")
    L.append("")
    L.append("---")
    L.append("")

    # 详细结果
    L.append("## 逐步结果")
    L.append("")
    for s in res.steps:
        L.append(f"### 步骤 {s.index}　{s.step.title}")
        L.append("")
        L.append(s.step.detail)
        L.append("")
        if s.ok:
            body = _render_payload(s.payload, s.step.kind)
            if body:
                L.append(body)
                L.append("")
            if s.step.note:
                L.append(f"> {s.step.note}")
                L.append("")
        else:
            L.append("**执行失败**：")
            L.append("")
            L.append("```")
            L.append(s.error.strip()[:1500])
            L.append("```")
            L.append("")

    if m.get("conclusion"):
        L.append("---")
        L.append("")
        L.append("## 结论")
        L.append("")
        L.append(m["conclusion"])
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*本报告由 `code/run.py` 自动生成。复现代码见 `code/repl/{m['id'].lower()}_*.py`，"
             f"公共工具见 `code/common/`。所有数据为合成数据，结论仅用于方法学验证。*")
    return "\n".join(L)


def build_reports_index(all_results: list[RunResult]) -> str:
    L: list[str] = []
    L.append("# 复现报告索引")
    L.append("")
    L.append(f"共 **{len(all_results)}** 篇复现，"
             f"合计 **{sum(len(r.steps) for r in all_results)}** 个步骤，"
             f"成功 **{sum(r.ok_count for r in all_results)}** 步。")
    L.append("")
    L.append("| 编号 | 论文 | 期刊 / 年份 | 位次 | 步骤 | 耗时 | 报告 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        m = r.meta
        status = "全部通过" if not r.failed else f"**{len(r.failed)} 步失败**"
        L.append(f"| {m['id']} | {m['title']} | {m['journal']} | {m['position']} | "
                 f"{r.ok_count}/{len(r.steps)} | {r.total_seconds:.1f}s | "
                 f"[{m['id']}]({m['id']}-{m.get('slug', 'report')}.md) |")
    L.append("")
    L.append(f"*生成时间：{datetime.now():%Y-%m-%d %H:%M}*")
    return "\n".join(L)


def write_report(res: RunResult) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = res.meta.get("slug") or res.meta["id"]
    path = REPORTS_DIR / f"{res.meta['id']}-{slug}.md"
    path.write_text(build_report(res), encoding="utf-8")
    return path


def write_paper_guide(res: RunResult) -> Path:
    """生成「逐步引导」文档：读者照着做的操作手册。"""
    m = res.meta
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append(f"# {m['id']}　{m['title']}")
    L.append("")
    L.append(f"**原文**：{m['journal']}"
             + (f"（DOI: {m['doi']}）" if m.get("doi") else "")
             + f"　**作者位次**：{m['position']}")
    L.append("")
    L.append(f"**复现目标**：{m['goal']}")
    L.append("")
    L.append(f"**与原论文的差异**：{m['difference']}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 你会学到什么")
    L.append("")
    for point in m.get("learn", []):
        L.append(f"- {point}")
    L.append("")
    L.append("## 逐步引导")
    L.append("")
    L.append("```bash")
    L.append(f"cd yang-lab-replications")
    L.append(f"python code/run.py --id {m['id']}")
    L.append("```")
    L.append("")
    L.append("运行后终端会打印每一步的结果，并在 `reports/` 生成完整报告。")
    L.append("")
    for i, s in enumerate(_steps_of(m["id"]), 1):
        L.append(f"### 第 {i} 步　{s.title}")
        L.append("")
        L.append(s.detail)
        L.append("")
        if s.note:
            L.append(f"> {s.note}")
            L.append("")
        L.append("**你应该看到什么**：见 [报告](../reports/) 中对应章节。")
        L.append("")
    L.append("## 关键代码")
    L.append("")
    L.append(f"- 复现逻辑：`code/repl/{m['id'].lower()}_*.py`")
    L.append("- 公共工具：`code/common/`")
    L.append("")
    L.append("## 延伸练习")
    L.append("")
    for ex in m.get("exercises", []):
        L.append(f"- {ex}")
    L.append("")
    path = PAPERS_DIR / f"{m['id']}-{m.get('slug', 'guide')}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _steps_of(rid: str) -> list[Step]:
    for fid, mod in discover():
        if fid == rid:
            return mod.steps()
    return []
