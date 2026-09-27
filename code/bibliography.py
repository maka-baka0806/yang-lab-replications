#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 BIBLIOGRAPHY.md（完整文献库，含复现映射）。"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import framework as fw  # noqa: E402
from common.bibliography_full import (AWARDS, CONFERENCE_ABSTRACTS, EARLY_PHYSICS,  # noqa: E402
                                      JOURNAL_PAPERS, PREPRINTS, THESES_AND_PROJECTS,
                                      stats)


def build() -> str:
    s = stats()
    mods = {m.META.get("doi", ""): rid for rid, m in fw.discover() if m.META.get("doi")}
    slug = {rid: m.META.get("slug", "") for rid, m in fw.discover()}

    L = ["# 杨振宇老师 · 完整文献库", "",
         f"> 共 **{s['合计']}** 条，逐条经 API 核验并排除同名研究者。"
         f"数据来源见 `核查记录/complete_publication_list.md`。", "",
         "| 类别 | 数量 |", "|---|---|"]
    for k in ["期刊论文", "会议摘要", "预印本", "学位论文/项目", "早期物理研究"]:
        L.append(f"| {k} | {s[k]} |")
    L += [f"| **合计** | **{s['合计']}** |", "",
          f"其中第一作者（★）**{s['第一作者(★)']}** 篇，通讯/末位（☆）**{s['通讯末位(☆)']}** 篇。", "",
          "---", ""]

    def table(rows, with_link=True):
        L.append("| 年份 | 位次 | 标题 | 出处 | 复现 |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            rid = mods.get(r.get("doi", ""))
            link = f"[{rid}](reports/{rid}-{slug.get(rid,'')}.md)" if rid else "—"
            title = r["title"]
            if r.get("doi"):
                title = f"[{title[:72]}](https://doi.org/{r['doi']})"
            pos_raw = r["position"] or ""
            pos = next((x for x in ("★", "☆", "○") if x in pos_raw), "—")
            L.append(f"| {r['year']} | {pos} | {title} | {r['journal']} | {link if with_link else '—'} |")
        L.append("")

    L += ["## A. 期刊论文（医学物理 / 医学影像）", ""]
    table(JOURNAL_PAPERS)
    L += ["## B. 会议摘要（AAPM / ASTRO-RSS）", "",
          "> AAPM 年会摘要发表于 *Medical Physics* 摘要期，ASTRO 摘要发表于 *IJROBP* 增刊。"
          "全部经 Scholars@Duke 出版记录逐条核验作者列表。", ""]
    table(CONFERENCE_ABSTRACTS, with_link=False)
    L += ["## C. 预印本", ""]
    table(PREPRINTS, with_link=False)
    L += ["## D. 学位论文与科研项目", ""]
    table(THESES_AND_PROJECTS, with_link=False)
    L += ["**获奖**", ""]
    for a in AWARDS:
        L.append(f"- {a}")
    L += ["", "## E. 早期物理研究（2019–2022，东南大学 侯吉旋 组）", ""]
    table(EARLY_PHYSICS, with_link=False)
    L += ["---", "",
          "*本文件由 `code/bibliography.py` 自动生成。*"]
    return "\n".join(L)


if __name__ == "__main__":
    out = ROOT.parent / "BIBLIOGRAPHY.md"
    out.write_text(build(), encoding="utf-8")
    print(f"已写入 {out}")
