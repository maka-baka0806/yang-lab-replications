#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
覆盖矩阵：把「完整文献库」与「已实现的复现」对照起来。
    python code/coverage.py --write     # 写入 COVERAGE.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import framework as fw  # noqa: E402
from common.bibliography_full import (CONFERENCE_ABSTRACTS, EARLY_PHYSICS,  # noqa: E402
                                      JOURNAL_PAPERS, PREPRINTS,
                                      THESES_AND_PROJECTS, stats)

# 需要真实数据/大规模训练，无法用合成数据复现的条目 → 原因
NO_REPL_REASON = {
    "Radiogenomic explainable AI": "需真实基因组数据与活检病理标签",
    "An Explainable Deep Model for Risk Scoring": "同上，且依赖完整 LRP + 决策场重构",
    "Explainable Artificial Intelligence Techniques in Medical Imaging Analysis":
        "学位论文，其三个研究章节分别由 R01 / R08 / R04 覆盖",
    "Development of a Voxel-Based Radiomics Calculation Platform":
        "硕士论文，其内容由 R01 的滤波框架覆盖",
    "放疗中腹部图像自动分割": "科研项目，方法学由 R02 覆盖",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    mods = fw.discover()
    by_doi = {m.META.get("doi", ""): rid for rid, m in mods if m.META.get("doi")}
    slug = {rid: m.META.get("slug", "") for rid, m in mods}
    topic_hit = {  # 标题关键词 → 复现编号（用于无 DOI 的会议摘要/预印本）
        "Radiomic Sequence": "R09", "Pulmonary Ventilation": "R09",
        "Spherical Slicing": "R03", "SIMT": "R21", "Single-Isocenter": "R03",
        "Vision Transformer": "R18", "Mammography": "R19", "Synthetic Poisoning": "R20",
        "Dosiomic": "R17", "Dosiomics-Based": "R17", "Radiogenomic": "R16",
        "Glioblastoma Post-Resection": "R05", "Discretization": "R10",
        "Radionecrosis": "R16", "Dual-Radiomics": "R05", "Dual Radiomic": "R17",
        "Dose-Incorporated": "R14", "Kolmogorov": "R15", "Neural Ordinary": "R08",
        "Radiomics": "R01", "Uncertainty": "R02", "Swin": "R19", "Synthetic CT": "R20",
        "U-Net": "R20", "Dose Map": "R13", "DVH": "R06", "PhysMorph": "R07",
    }

    def find_rep(title: str, doi: str) -> str:
        if doi and doi in by_doi:
            rid = by_doi[doi]
            return f"[{rid}](reports/{rid}-{slug.get(rid,'')}.md)"
        for k, rid in topic_hit.items():
            if k.lower() in title.lower():
                return f"~{rid}"
        return "—"

    L = ["# 文献覆盖矩阵", "",
         f"文献库共 **{stats()['合计']}** 条；已实现 **{len(mods)}** 个可执行复现"
         f"（合计 **{sum(len(m.steps()) for _, m in mods)}** 个步骤）。", "",
         "> `[Rxx]` = 有独立复现报告　`~Rxx` = 相关复现覆盖其方法　`—` = 未复现（原因见后）", ""]

    for name, rows, note in [
        ("A. 期刊论文", JOURNAL_PAPERS, ""),
        ("B. 会议摘要", CONFERENCE_ABSTRACTS, "会议摘要多为期刊全文的阶段性成果，方法由对应复现覆盖"),
        ("C. 预印本", PREPRINTS, ""),
        ("D. 学位论文与科研项目", THESES_AND_PROJECTS, ""),
        ("E. 早期物理研究", EARLY_PHYSICS, "统计物理/生物物理方向，不适用于医学影像复现框架"),
    ]:
        L += [f"## {name}", ""]
        if note:
            L += [f"> {note}", ""]
        L += ["| 年份 | 位次 | 标题 | 出处 | 复现 |", "|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: (-int(x["year"]), x["title"])):
            pos = next((x for x in ("★", "☆", "○") if x in (r.get("position") or "")), "—")
            rep = "—" if name.startswith("E") else find_rep(r["title"], r.get("doi", ""))
            L.append(f"| {r['year']} | {pos} | {r['title'][:70]} | {r['journal'][:34]} | {rep} |")
        L.append("")

    L += ["## 未实现可执行复现的条目及原因", "", "| 条目 | 原因 |", "|---|---|"]
    for r in JOURNAL_PAPERS + THESES_AND_PROJECTS:
        if by_doi.get(r.get("doi", "")):
            continue
        why = next((v for k, v in NO_REPL_REASON.items() if k in r["title"]), None)
        if why:
            L.append(f"| {r['title'][:64]} | {why} |")
    L += ["", "---", "", "*由 `code/coverage.py` 自动生成。*"]

    text = "\n".join(L)
    if a.write:
        out = ROOT.parent / "COVERAGE.md"
        out.write_text(text, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        print(text[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
