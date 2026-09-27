# 杨振宇老师论文复现项目

> 把昆山杜克大学医学物理 **杨振宇** 老师已发表论文的方法学，**逐步骤复现成可运行的代码**，
> 并为每一篇生成一份**带实测数据的复现报告**。

这不是文献综述，也不是代码仓库 —— 是一套**可以照着做、做完有产出**的复现工程。

---

## 这是什么

| 你想要的 | 这个项目给的 |
|---|---|
| 看懂他的论文 | 每篇一份 `papers/` **逐步引导**：告诉你每一步做什么、对应原文哪个环节 |
| 亲手做一遍 | `code/repl/` 里每篇一个**可执行模块**，跑一条命令就出结果 |
| 有产出 | 每篇自动生成 `reports/` **实测报告**：带真实数字的表格与结论对照 |
| 知道差在哪 | 每篇都标注**「与原论文的差异」**：哪些是合成数据替代、哪些只能比较方向 |

---

## 快速开始

```bash
cd ~/医学影像学工作/yang-lab-replications
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1

# 看有哪些复现
python code/run.py --list

# 跑一篇（先看终端输出）
python code/run.py --id R01 --stdout

# 跑一篇并生成报告
python code/run.py --id R01

# 跑全部并生成报告与索引
python code/run.py --all
```

解释器用 `~/miniconda3/envs/medimg/bin/python`（或先 `conda activate medimg`）。

---

## 目录结构

```
yang-lab-replications/
├── README.md              # 本文件：总览与覆盖矩阵
├── RUNBOOK.md             # 逐步操作手册（新手照做）
├── requirements.txt
├── code/
│   ├── framework.py       # 框架：契约 + 运行器 + 报告生成器
│   ├── run.py             # 入口命令
│   ├── common/            # 公共工具（体模生成、指标、滤波、配准…）
│   └── repl/              # 每篇论文一个复现模块 rXX_*.py
├── papers/                # 逐步引导（自动生成）
└── reports/               # 实测报告（自动生成）
```

---

## 覆盖矩阵

见 [reports/README.md](reports/README.md)（每次 `--all` 后自动更新）。
每篇复现包含 **5–7 个可独立执行的步骤**，全部使用合成数据，单篇耗时通常在 **1–15 秒**。

### 为什么用合成数据

论文用的是 VAMPIRE / BraTS / TCIA Lung1 等**真实临床数据集**，需要申请与伦理审批。
本项目的做法是：**造出保留了关键难点的合成数据**（如同一位置的强度不均匀、气体潴留、
多靶点几何、形变场），从而把论文的**方法学链条**完整跑通。

因此：
- **能比较的**：方法方向、结论方向、指标间的关系、参数敏感性
- **不能比较的**：绝对数值（如 Dice 0.78、AUC 0.92 这类具体数字）

每一篇的引导与报告里都写明了这一点。

---

## 三类读者怎么用

**A. 想快速了解他做了什么** → 先读 `reports/README.md` 的覆盖矩阵，再挑 2–3 篇报告看结论章节。

**B. 想学会方法** → 按 `RUNBOOK.md` 走，每篇先读 `papers/RXX-*.md` 的逐步引导，再跑代码。

**C. 想改代码做自己的研究** → 直接看 `code/repl/rXX_*.py`，每篇 5–7 个函数，注释对应原文小节。

---

## 工程约定

每个复现模块只暴露两个东西：

```python
META = dict(id="R01", year=2022, title=..., journal=..., doi=...,
            position="★ 第一作者", goal=..., difference=...,
            conclusion=..., learn=[...], exercises=[...])

def steps() -> list[Step]:   # Step(title, detail, fn, kind, note)
    ...
```

- `fn(ctx)` 接收共享状态字典，可把中间结果存进 `ctx` 给后续步骤用
- `kind ∈ {"text", "metrics", "table"}`
- 步骤必须**确定性可复现**：所有随机过程固定种子

新增一篇复现 = 新建一个 `code/repl/rXX_*.py`，不需要改框架。

---

## 依赖

```
numpy, scipy, pandas, scikit-learn, scikit-image,
SimpleITK, torch, pyradiomics（可选，部分模块需要）
```

---

## 说明

- 作者的文献归属经过**逐个核对**（排除同名研究者），位次标注：
  ★ 第一作者 ｜ ☆ 通讯/末位（导师位）｜ ○ 合作者
- 复现代码为**独立实现**，不是原文作者提供的代码；原文的 MATLAB 自研工具箱等
  以等价的向量化实现替代，并在报告中注明
- 本项目的定位是**教学与方法学验证**，不能用于临床决策
