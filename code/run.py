#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现运行器
============
    python code/run.py --list            列出全部复现
    python code/run.py --all             运行全部 + 生成报告与引导
    python code/run.py --id R01          运行单篇
    python code/run.py --id R01 --stdout 只打印，不写文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import framework as fw  # noqa: E402


def _print_result(res: fw.RunResult) -> None:
    m = res.meta
    print(f"\n{'=' * 72}")
    print(f"  {m['id']} · {m['title']}")
    print(f"  {m['journal']}　·　{m['position']}")
    print(f"{'=' * 72}")
    for s in res.steps:
        mark = "OK  " if s.ok else "FAIL"
        print(f"  [{mark}] 步骤 {s.index}  {s.step.title}   ({s.seconds:.2f}s)")
        if not s.ok:
            print(f"         {s.error.splitlines()[0]}")
    print(f"  ---------------------------------------------")
    print(f"  合计 {res.ok_count}/{len(res.steps)} 步通过，"
          f"耗时 {res.total_seconds:.1f} 秒")


def main() -> int:
    ap = argparse.ArgumentParser(description="杨振宇老师论文复现运行器")
    ap.add_argument("--list", action="store_true", help="列出全部复现")
    ap.add_argument("--all", action="store_true", help="运行全部")
    ap.add_argument("--id", help="只运行指定编号，如 R01")
    ap.add_argument("--stdout", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    modules = fw.discover()
    if not modules:
        print("未发现任何复现模块（code/repl/ 下需有 META 与 steps）")
        return 1

    if args.list:
        print(f"共 {len(modules)} 篇复现：\n")
        for rid, mod in modules:
            m = mod.META
            n = len(mod.steps())
            print(f"  {rid}  {m['year']}  {m['title'][:52]:54s} {n} 步")
        return 0

    targets = modules if args.all else [x for x in modules if x[0] == args.id]
    if not targets:
        print(f"未找到 {args.id}。可用：{', '.join(r for r, _ in modules)}")
        return 1

    results: list[fw.RunResult] = []
    for rid, mod in targets:
        res = fw.run_module(mod, progress=lambda s: print(s))
        _print_result(res)
        results.append(res)
        if not args.stdout:
            p = fw.write_report(res)
            g = fw.write_paper_guide(res)
            print(f"  报告已写入：{p.relative_to(ROOT.parent)}")
            print(f"  引导已写入：{g.relative_to(ROOT.parent)}")

    if not args.stdout and len(results) > 1:
        idx = fw.REPORTS_DIR / "README.md"
        idx.write_text(fw.build_reports_index(results), encoding="utf-8")
        print(f"\n索引已写入：{idx.relative_to(ROOT.parent)}")

    total_fail = sum(len(r.failed) for r in results)
    print(f"\n{'=' * 72}")
    print(f"  总计：{len(results)} 篇，"
          f"{sum(r.ok_count for r in results)}/{sum(len(r.steps) for r in results)} 步通过")
    print(f"{'=' * 72}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
