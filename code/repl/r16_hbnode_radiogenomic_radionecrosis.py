"""
R16 · 影像基因组学 HBNODE：脑转移放射性坏死 vs 肿瘤复发
=========================================================
原文：Zhao J, Vaios E, **Yang Z**, Lu K, Floyd S, Yang D, Ji H, Reitman ZJ,
      Lafata KJ, Fecci P, Kirkpatrick JP, Wang C. Radiogenomic explainable AI with
      neural ordinary differential equation for identifying post-SRS brain metastasis
      radionecrosis. Med Phys 2025;52(4):2661-2674.

论文的核心主张
--------------
1. 把**影像深度特征 + 基因组标志物 + 临床参数**融合进同一个潜空间，
   解决「放射性坏死 vs 肿瘤复发」这一传统上要靠活检才能鉴别的问题；
2. 假设深度特征提取可以建模为**时空连续过程**，用 HBNODE（Heavy-Ball 神经 ODE）
   描述样本在潜空间里「走向诊断结论」的轨迹：
        dz/dt = v,   dv/dt = −γ v − ∇U(z) + f(z, t)
   其中 −γv 是阻尼、−∇U 是势能梯度（重球动量项）、f 是神经网络驱动力；
3. 于是可以把每个病例的诊断轨迹导出/可视化，把黑箱变成可观察的动力系统。

本复现做什么
------------
造一个 n=220 的三模态合成队列（影像深度特征 32 维 / 基因组 4 个标志物 / 临床 3 项），
三类特征由三个**样本内正交**的潜在因子驱动、结局由三者共同决定（单源都不完整）。
用手写 RK4 积分器实现 HBNODE（不依赖 torchdiffeq），
报告判别性能、潜空间诊断轨迹的分离度随演化时间的变化，
再做多模态消融与「逐组置换」的重要性分析（对应原文 LRP 的可解释性精神）。
所有 AUC 都是 3 次独立分层划分的测试集结果均值。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework import Step
from common.modeling import binary_metrics

# 预热 sklearn.metrics：它的首次导入在负载高的机器上要好几秒，
# 放在模块导入期（不计入任何步骤耗时），避免把导入开销算到步骤里。
import sklearn.metrics  # noqa: F401

# ----------------------------------------------------------------------
# 实验常量：规模刻意压小，保证每一步都能在十几秒内跑完
# ----------------------------------------------------------------------
N_SAMPLES = 220
N_IMAGING = 32           # 影像深度特征维度
N_SIGNAL_DIM = 12        # 其中真正携带信号的维度（其余是纯噪声，模拟冗余深度特征）
N_GENOMIC = 4            # EGFR / ALK / KRAS / PD-L1
N_CLINICAL = 3           # 年龄 / 处方剂量 / 肿瘤体积
PREVALENCE = 0.45
TEST_FRAC = 0.30
SPLIT_SEEDS = (0, 1, 2)  # 3 次独立分层划分，报告均值 ± 标准差
DIM = 6                  # 潜空间维度（z 与 v 各 6 维）
HIDDEN = 16              # 驱动力 f 的隐层宽度
RK4_STEPS = 4            # RK4 积分步数（t: 0 → 1）
EPOCHS = 40
LR = 1e-2
WEIGHT_DECAY = 1e-3
N_TRAJ = 9               # 轨迹记录的时间点数
SEED = 0

GENOMIC_COLS = ["EGFR", "ALK", "KRAS", "PD-L1"]
CLINICAL_COLS = ["年龄", "处方剂量(Gy)", "肿瘤体积(cc)"]


# ----------------------------------------------------------------------
# HBNODE：重球动力学 + 手写 RK4
# ----------------------------------------------------------------------
class HBNODE(nn.Module):
    """Heavy-Ball 神经 ODE 分类器。

    状态 (z, v) ∈ R^d × R^d，动力学为
        dz/dt = v
        dv/dt = −softplus(log γ)·v − softplus(log a)⊙z + f(z, t)
    softplus 保证阻尼 γ > 0；势能 U(z) = ½Σ a_i z_i² 是凸的（∇U = a⊙z），
    因此 −∇U 是稳定的回复力，f 提供数据驱动的非保守力。
    """

    def __init__(self, n_in: int, dim: int = DIM, hidden: int = HIDDEN):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 32), nn.Tanh(), nn.Linear(32, 2 * dim))
        self.f = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim))
        self.log_a = nn.Parameter(torch.zeros(dim))
        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(dim, 2)

    def deriv(self, z, v, t):
        grad_u = F.softplus(self.log_a) * z
        damp = F.softplus(self.log_gamma)
        tt = torch.full((z.shape[0], 1), float(t), dtype=z.dtype)
        return v, -damp * v - grad_u + self.f(torch.cat([z, tt], dim=1))

    def rk4(self, z, v, t0: float, t1: float, n_steps: int = RK4_STEPS):
        """经典四阶 Runge–Kutta：每步 4 次右端项求值，误差 O(h⁵)。"""
        h = (t1 - t0) / n_steps
        t = t0
        for _ in range(n_steps):
            k1z, k1v = self.deriv(z, v, t)
            k2z, k2v = self.deriv(z + 0.5 * h * k1z, v + 0.5 * h * k1v, t + 0.5 * h)
            k3z, k3v = self.deriv(z + 0.5 * h * k2z, v + 0.5 * h * k2v, t + 0.5 * h)
            k4z, k4v = self.deriv(z + h * k3z, v + h * k3v, t + h)
            z = z + h * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
            v = v + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            t = t + h
        return z, v

    def forward(self, x, times: np.ndarray | None = None):
        h = self.encoder(x)
        z, v = h[:, :self.dim], h[:, self.dim:]
        if times is None:
            z, _ = self.rk4(z, v, 0.0, 1.0)
            return self.head(z)
        traj, t_prev = [], 0.0
        for t in times:
            if t > t_prev:
                z, v = self.rk4(z, v, t_prev, float(t))
                t_prev = float(t)
            traj.append(z)
        return self.head(z), torch.stack(traj, dim=0)


# ----------------------------------------------------------------------
# 训练与评估
# ----------------------------------------------------------------------
def train_hbnode(Xtr, ytr, Xte, yte, epochs: int = EPOCHS, lr: float = LR,
                 wd: float = WEIGHT_DECAY, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = HBNODE(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            hist.append({"epoch": ep, "训练损失": round(float(loss.detach()), 4)})
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte), dim=1)[:, 1].numpy()
        train_acc = float((model(Xtr).argmax(1) == ytr).float().mean())
    return {"model": model, "prob": prob, "yte": yte.numpy(),
            "metrics": binary_metrics(yte.numpy(), prob), "history": hist,
            "耗时 (s)": round(elapsed, 3), "训练准确率": round(train_acc, 4),
            "参数量": int(sum(p.numel() for p in model.parameters()))}


def _stratified_split(y: np.ndarray, seed: int, frac: float = TEST_FRAC):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for cls in (0, 1):
        ids = np.where(y == cls)[0].copy()
        rng.shuffle(ids)
        k = int(round(len(ids) * frac))
        te.extend(ids[:k])
        tr.extend(ids[k:])
    return np.sort(np.array(tr)), np.sort(np.array(te))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 相关（用秩的 Pearson 相关实现，避免额外导入）。"""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().values
    rb = pd.Series(np.asarray(b, dtype=float)).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------
# META
# ----------------------------------------------------------------------
META = dict(
    id="R16",
    year=2025,
    title="影像基因组学 HBNODE：脑转移放射性坏死 vs 肿瘤复发",
    journal="Med Phys 2025;52(4):2661-2674",
    doi="10.1002/mp.17635",
    position="○ 合作者（第 3 / 12 作者）",
    slug="radiogenomic-hbnode-radionecrosis",
    goal="把影像深度特征、基因组标志物与临床参数融合进一个 Heavy-Ball 神经 ODE 潜空间，"
         "复现「深度特征提取可建模为时空连续过程」这一假设，并验证三模态融合的判别能力"
         "优于任何单模态，同时导出可解释的潜空间诊断轨迹。",
    difference="论文用真实脑转移 MRI 的深度特征与真实基因组 panel；本复现用三模态合成队列"
               "（32 维合成深度特征 + 4 个合成基因组标志物 + 3 项合成临床参数），"
               "模型被压到 6 维潜空间、40 个 epoch、CPU 训练，只报告 3 次分层划分的测试集 AUC。"
               "因此只比较「融合 vs 单模态」的方法学方向与轨迹可解释性，不比较绝对 AUC，"
               "也不复现原文的 LRP 归因图。",
    conclusion=(
        "在三模态合成队列上，HBNODE 把 39 维输入编码到 6 维潜空间后，用重球动力学"
        "（阻尼 −γv、凸势能梯度 −a⊙z、神经驱动力 f(z,t)）经 RK4 积分到 t=1，"
        "3 次分层划分的平均测试 AUC 为 0.770（0.769 / 0.707 / 0.833），"
        "高于影像+基因组 0.729、影像+临床 0.641 与仅影像 0.601 —— 与原文"
        "「影像 + 基因组 + 临床三源融合才能可靠鉴别放射性坏死与肿瘤复发」的结论方向一致，"
        "且逐组置换显示基因组标志物的贡献最大（整组打乱后 AUC 下降 0.135，其中 EGFR 独占 0.062），"
        "影像深度特征次之（0.086），这与原文把基因组标志物作为核心变量的定位吻合。"
        "潜空间轨迹方面得到一个**部分吻合**的结果：两类样本的质心距离随演化时间 t 从 1.48 "
        "单调增大到 4.08（×2.76），说明动力学确实在一个连续时间里持续改变样本状态；"
        "但 Fisher 分离比（2.62 → 2.60）与「把分类头接到中间状态」的 AUC（0.771 → 0.769）"
        "几乎不变 —— 也就是本复现里判别信息主要在编码器里一次成型，ODE 演化承担的是整体"
        "输运/扩张而不是「逐步推理」。因此可以说连续过程的**存在性**得到了验证，"
        "但在这个简化设定下它并不是判别力的主要来源；要复现原文更强的轨迹解释性，"
        "需要限制编码器容量并把监督信号加到整条轨迹上。"
        "需要强调：合成数据的绝对 AUC 没有临床意义，本复现能带走的结论是"
        "「HBNODE 的多模态融合成立，潜空间轨迹可导出、可观察」。"),
    learn=[
        "Heavy-Ball 动力学如何写成一个 ODE：dz/dt=v、dv/dt=−γv−∇U(z)+f(z,t)，以及为什么用 softplus 保证 γ>0、U 凸",
        "手写 RK4 积分器：四阶精度、每步 4 次右端项求值，以及用「步长减半」验证积分器收敛性的做法",
        "把「样本走向结论」写成潜空间轨迹后，如何用类间质心距离 / Fisher 比量化「结论何时形成」",
        "多模态消融实验设计：每加一个模态看 AUC 增量，比只报告融合模型的 AUC 更有说服力",
        "置换重要性（逐组打乱）如何在不打开黑箱的前提下回答「哪个模态不可替代」",
    ],
    exercises=[
        "把 RK4_STEPS 从 4 改成 1（退化成欧拉法），观察 AUC 与轨迹平滑度的退化",
        "把 log_gamma 的参数化去掉（固定 γ=0 或 γ=100），比较无阻尼与过阻尼两种极限下的分离速度",
        "把影像深度特征的信噪比系数从 0.7 调到 0.25，看融合相对单模态的增益是否变大",
    ],
)


# ----------------------------------------------------------------------
# 步骤
# ----------------------------------------------------------------------
def steps() -> list[Step]:
    def s1(ctx):
        """造三模态合成队列：三类特征由三个样本内正交的潜在因子驱动。"""
        rng = np.random.RandomState(20240517)
        latent = rng.randn(N_SAMPLES, 3)
        latent -= latent.mean(axis=0, keepdims=True)
        latent = np.linalg.qr(latent)[0] * np.sqrt(N_SAMPLES)   # 样本内正交
        li, lg, lc = latent[:, 0], latent[:, 1], latent[:, 2]

        # 影像深度特征：只有前 N_SIGNAL_DIM 维带信号，其余是冗余噪声
        img = rng.randn(N_SAMPLES, N_IMAGING)
        img[:, :N_SIGNAL_DIM] = 0.70 * li[:, None] + rng.randn(N_SAMPLES, N_SIGNAL_DIM)

        geno = pd.DataFrame({
            "EGFR": 0.9 * lg + rng.randn(N_SAMPLES) * 0.5,
            "ALK": (lg + rng.randn(N_SAMPLES) * 0.7 > 0.4).astype(float),
            "KRAS": (lg + rng.randn(N_SAMPLES) * 0.7 > 0.9).astype(float),
            "PD-L1": 0.9 * lg + rng.randn(N_SAMPLES) * 0.5,
        })
        clin = pd.DataFrame({
            "年龄": 62 + 9 * lc + rng.randn(N_SAMPLES) * 3,
            "处方剂量(Gy)": 20 + 2.2 * lc + rng.randn(N_SAMPLES) * 0.8,
            "肿瘤体积(cc)": 5 + 1.6 * lc + rng.randn(N_SAMPLES) * 1.1,
        })

        risk = 0.95 * latent.sum(axis=1) + 0.9 * rng.randn(N_SAMPLES)
        y = (risk >= np.quantile(risk, 1 - PREVALENCE)).astype(int)

        X_img = pd.DataFrame(img, columns=[f"deep_{i+1:02d}" for i in range(N_IMAGING)])
        X = pd.concat([X_img, geno, clin], axis=1)
        ctx.update(X=X, y=y, X_img=X_img, geno=geno, clin=clin, latent=latent,
                   X_img_cols=list(X_img.columns),
                   splits=[_stratified_split(y, s) for s in SPLIT_SEEDS],
                   split_seeds=list(SPLIT_SEEDS))

        rows = []
        for name, cols in [("影像深度特征", list(X_img.columns)),
                           ("基因组标志物", GENOMIC_COLS),
                           ("临床参数", CLINICAL_COLS)]:
            rhos = {c: abs(_spearman(X[c].values, y)) for c in cols}
            top = max(rhos, key=rhos.get)
            rows.append({"模态": name, "特征数": len(cols),
                         "最强单特征": top, "|Spearman ρ|": round(rhos[top], 3)})
        rows.append({"模态": "结局（放射性坏死=1）", "特征数": int(y.sum()),
                     "最强单特征": f"阳性率 {y.mean():.2f}",
                     "|Spearman ρ|": "-"})
        return pd.DataFrame(rows)

    def s2(ctx):
        """实现 HBNODE（重球动力学 + 手写 RK4）并检查结构与积分器精度。"""
        n_in = N_IMAGING + N_GENOMIC + N_CLINICAL
        net = HBNODE(n_in)
        torch.manual_seed(0)
        z = torch.randn(16, DIM)
        v = torch.randn(16, DIM)
        z1 = net.rk4(z, v, 0.0, 1.0, n_steps=RK4_STEPS)[0]
        z2 = net.rk4(z, v, 0.0, 1.0, n_steps=2 * RK4_STEPS)[0]
        z0 = net.rk4(z, v, 0.0, 1.0, n_steps=1)[0]
        dev_rk4 = float((z1 - z2).detach().abs().max())
        dev_euler = float((z1 - z0).detach().abs().max())
        return {
            "潜空间维度 d（z 与 v 各 d 维）": DIM,
            "驱动力 f 的隐层宽度": HIDDEN,
            "RK4 步数（t: 0→1）": RK4_STEPS,
            "编码器参数量": int(sum(p.numel() for p in net.encoder.parameters())),
            "动力学 f 参数量": int(sum(p.numel() for p in net.f.parameters())),
            "模型总参数量（39 维输入）": int(sum(p.numel() for p in net.parameters())),
            "阻尼 γ 初值 softplus(0)": round(float(F.softplus(torch.tensor(0.0))), 4),
            "势能系数 a 初值 softplus(0)": round(float(F.softplus(torch.tensor(0.0))), 4),
            "RK4(4 步) vs RK4(8 步) 末态最大偏差": f"{dev_rk4:.2e}",
            "单步欧拉 vs RK4(4 步) 末态最大偏差": f"{dev_euler:.2e}",
        }

    def s3(ctx):
        """在 3 次分层划分上训练 HBNODE（三模态输入），报告判别性能。"""
        torch.set_num_threads(1)
        Xnp, y = ctx["X"].values, ctx["y"]
        runs = []
        for tr, te in ctx["splits"]:
            runs.append(train_hbnode(
                torch.tensor(Xnp[tr], dtype=torch.float32), torch.tensor(y[tr]),
                torch.tensor(Xnp[te], dtype=torch.float32), torch.tensor(y[te])))
        ctx["fused_runs"] = runs
        ctx["main"] = runs[0]
        ctx["tr0"], ctx["te0"] = ctx["splits"][0]

        aucs = [r["metrics"]["AUC"] for r in runs]
        accs = [r["metrics"]["准确率"] for r in runs]
        sens = [r["metrics"]["灵敏度"] for r in runs]
        spec = [r["metrics"]["特异度"] for r in runs]
        ctx["fused_summary"] = {"AUC": float(np.mean(aucs))}
        return {
            "训练 / 测试样本数": f"{len(ctx['tr0'])} / {len(ctx['te0'])} × {len(runs)} 次划分",
            "输入维度（三模态）": int(Xnp.shape[1]),
            "模型参数量": runs[0]["参数量"],
            "训练轮数": EPOCHS,
            "单次训练耗时 (s)": runs[0]["耗时 (s)"],
            "测试 AUC（3 次划分）": " / ".join(f"{a:.3f}" for a in aucs),
            "测试 AUC 均值 ± 标准差": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
            "准确率 均值": round(float(np.mean(accs)), 4),
            "灵敏度（放射性坏死）均值": round(float(np.mean(sens)), 4),
            "特异度（肿瘤复发）均值": round(float(np.mean(spec)), 4),
            "训练集准确率（3 次划分均值）": round(
                float(np.mean([r["训练准确率"] for r in runs])), 4),
        }

    def s4(ctx):
        """导出潜空间诊断轨迹：分离度与「中间状态判别能力」随演化时间的变化。"""
        model = ctx["main"]["model"]
        yte = ctx["main"]["yte"]
        Xte = torch.tensor(ctx["X"].values[ctx["te0"]], dtype=torch.float32)
        times = np.linspace(0.0, 1.0, N_TRAJ)
        model.eval()
        with torch.no_grad():
            _, traj = model(Xte, times=times)
            traj = traj.numpy()
            probs = torch.softmax(model.head(torch.tensor(traj)), dim=-1)[:, :, 1].numpy()

        rows = []
        for k, t in enumerate(times):
            zt = traj[k]
            c0, c1 = zt[yte == 0], zt[yte == 1]
            dist = float(np.linalg.norm(c0.mean(0) - c1.mean(0)))
            within = float(np.sqrt((c0.var(0).mean() + c1.var(0).mean()) / 2.0)) + 1e-9
            rows.append({
                "演化时间 t": round(float(t), 3),
                "类间质心距离": round(dist, 4),
                "类内散布": round(within, 4),
                "Fisher 分离比": round(dist / within, 4),
                "把分类头接到该时刻的 AUC": round(
                    float(binary_metrics(yte, probs[k])["AUC"]), 4),
            })
        ctx["traj"] = traj
        df = pd.DataFrame(rows)
        ctx["traj_table"] = df
        return df

    def s5(ctx):
        """多模态融合 vs 单模态消融：每加一个模态，测试 AUC 增加多少。"""
        Xnp, y = ctx["X"].values, ctx["y"]
        img_c = ctx["X_img_cols"]
        combos = [("仅影像", img_c),
                  ("影像 + 临床", img_c + CLINICAL_COLS),
                  ("影像 + 基因组", img_c + GENOMIC_COLS),
                  ("三模态全融合", img_c + CLINICAL_COLS + GENOMIC_COLS)]
        rows = []
        for label, cols in combos:
            if label == "三模态全融合":
                runs = ctx["fused_runs"]
            else:
                ci = [ctx["X"].columns.get_loc(c) for c in cols]
                runs = [train_hbnode(
                    torch.tensor(Xnp[tr][:, ci], dtype=torch.float32), torch.tensor(y[tr]),
                    torch.tensor(Xnp[te][:, ci], dtype=torch.float32), torch.tensor(y[te]))
                    for tr, te in ctx["splits"]]
            aucs = np.array([r["metrics"]["AUC"] for r in runs])
            rows.append({
                "输入组合": label, "输入维度": len(cols),
                "模型参数量": runs[0]["参数量"],
                "单次训练耗时 (s)": runs[0]["耗时 (s)"],
                "测试 AUC 均值": round(float(aucs.mean()), 4),
                "测试 AUC 标准差": round(float(aucs.std()), 4),
                "准确率 均值": round(float(np.mean(
                    [r["metrics"]["准确率"] for r in runs])), 4),
                "灵敏度 均值": round(float(np.mean(
                    [r["metrics"]["灵敏度"] for r in runs])), 4),
                "特异度 均值": round(float(np.mean(
                    [r["metrics"]["特异度"] for r in runs])), 4),
            })
        df = pd.DataFrame(rows)
        base = float(df.loc[df["输入组合"] == "仅影像", "测试 AUC 均值"].iloc[0])
        df["相对仅影像的 AUC 增量"] = np.round(df["测试 AUC 均值"] - base, 4)
        ctx["ablation_table"] = df
        return df

    def s6(ctx):
        """特征组置换重要性：逐组打乱输入，看 AUC 掉多少（LRP 的实用替代）。"""
        model = ctx["main"]["model"]
        model.eval()
        yte = ctx["main"]["yte"]
        Xte = torch.tensor(ctx["X"].values[ctx["te0"]], dtype=torch.float32)
        with torch.no_grad():
            base = float(binary_metrics(
                yte, torch.softmax(model(Xte), 1)[:, 1].numpy())["AUC"])

        cols = list(ctx["X"].columns)
        groups = [("影像深度特征（整组）", ctx["X_img_cols"]),
                  ("基因组标志物（整组）", GENOMIC_COLS),
                  ("临床参数（整组）", CLINICAL_COLS)] \
            + [(f"基因组 · {g}", [g]) for g in GENOMIC_COLS]

        rng = np.random.RandomState(0)
        rows = []
        for name, gc in groups:
            ci = [cols.index(c) for c in gc]
            drops = []
            for _ in range(12):
                Xp = Xte.clone()
                perm = torch.tensor(rng.permutation(Xp.shape[0]))
                Xp[:, ci] = Xte[perm][:, ci]
                with torch.no_grad():
                    p = torch.softmax(model(Xp), 1)[:, 1].numpy()
                drops.append(base - float(binary_metrics(yte, p)["AUC"]))
            rows.append({"置换对象": name,
                         "类型": "整个模态" if len(gc) > 1 else "单个标志物",
                         "AUC 下降（均值）": round(float(np.mean(drops)), 4),
                         "标准差": round(float(np.std(drops)), 4)})
        return pd.DataFrame(rows).sort_values(
            "AUC 下降（均值）", ascending=False).reset_index(drop=True)

    def s7(ctx):
        """结论对照：把复现结果与原文主张逐条对齐。"""
        ab = ctx["ablation_table"].set_index("输入组合")
        traj = ctx["traj_table"]
        fused = float(ab.loc["三模态全融合", "测试 AUC 均值"])
        singles = {k: float(ab.loc[k, "测试 AUC 均值"])
                   for k in ("仅影像", "影像 + 临床", "影像 + 基因组")}
        best = max(singles, key=singles.get)
        sep0 = float(traj["类间质心距离"].iloc[0])
        sep1 = float(traj["类间质心距离"].iloc[-1])
        fis0 = float(traj["Fisher 分离比"].iloc[0])
        fis1 = float(traj["Fisher 分离比"].iloc[-1])
        return {
            "主张 1 融合优于单模态": f"融合 {fused:.3f} vs 最佳缩减输入（{best}）{singles[best]:.3f}"
                                     f"（+{fused - singles[best]:.3f}）",
            "主张 2 深度特征提取是连续过程": f"类间质心距离 t=0→1：{sep0:.3f} → {sep1:.3f}"
                                             f"（×{sep1 / max(sep0, 1e-9):.2f}）",
            "Fisher 分离比 t=0→1": f"{fis0:.3f} → {fis1:.3f}（基本不变）",
            "主张 3 潜空间轨迹可解释": "轨迹与分离度已导出（步骤④表）；"
                                       "但判别力在 t=0 即已定型，演化主要起输运作用",
            "融合相对仅影像的增益": round(fused - singles["仅影像"], 4),
            "3 次划分 AUC 标准差": float(ab.loc["三模态全融合", "测试 AUC 标准差"]),
            "结论": "主张 1 方向一致；主张 2/3 部分吻合（连续演化存在，但判别力主要来自编码器）",
        }

    return [
        Step("① 造三模态合成队列（影像深度特征 + 基因组 + 临床）",
             "对应原文的数据构成。三个潜在因子（影像 / 基因组 / 临床）在样本内正交化，"
             "结局由三者等权驱动并按分位数二值化（阳性率 45%）；影像 32 维里只有 12 维带信号，"
             "模拟冗余的深度特征。同时准备 3 次独立分层划分供后面求均值。",
             s1, "table",
             "三源正交是刻意设计：这样「单模态 AUC 天花板」对三者相同，融合增益才有可比性。"),
        Step("② 实现 HBNODE（重球动力学 + 手写 RK4）",
             "对应原文的方法核心。编码器把输入映射到 (z₀, v₀)，再按 dz/dt=v、"
             "dv/dt=−γv−∇U(z)+f(z,t) 用 RK4 从 t=0 积分到 t=1，末态接线性分类头。"
             "本步检查结构与参数量，并用「步长减半的 RK4」和「单步欧拉」做积分器精度对照。",
             s2, "metrics",
             "softplus 保证阻尼 γ>0、势能 U(z)=½Σa_i z_i² 凸 —— −∇U 因此是稳定的回复力。"),
        Step("③ 训练 HBNODE 并报告判别性能",
             "对应原文的主结果。三模态输入（39 维）编码到 6 维潜空间，40 个 epoch、"
             "全批量 Adam、CPU 训练；在 3 次分层划分上各训练一次，报告测试 AUC 的均值与标准差。",
             s3, "metrics",
             "每次划分只有 66 例测试样本，单次 AUC 的标准误约 0.07，所以必须看 3 次的均值和标准差。"),
        Step("④ 导出潜空间诊断轨迹（分离度随演化时间变化）",
             "对应原文的轨迹可视化。记录 t = 0, 0.125, …, 1 共 9 个时刻的 z(t)，"
             "计算两类质心距离、类内散布与 Fisher 分离比；并把分类头接到每个中间状态上，"
             "看判别能力在演化的哪一段形成。",
             s4, "table",
             "这里要同时看两个指标：质心距离上升只说明状态在变，只有 Fisher 比上升才说明判别性在增强 —— "
             "本复现里前者上升、后者持平。"),
        Step("⑤ 多模态融合 vs 单模态消融",
             "对应原文的融合必要性论证。固定同一套 3 次划分，比较 仅影像 / 影像+临床 / "
             "影像+基因组 / 三模态全融合 四种输入的测试 AUC 与增量。",
             s5, "table",
             "「每加一个模态 AUC 增加多少」比只报告融合模型的 AUC 更能说明多模态的必要性。"),
        Step("⑥ 特征组置换重要性（可解释性）",
             "对应原文用 LRP 做的可解释性分析。在测试集上逐组打乱特征后重新预测，"
             "记录 AUC 的平均下降：三个模态各一组，基因组内部再逐个标志物做一次，"
             "回答「哪个来源、哪个标志物不可替代」。",
             s6, "table",
             "置换法只依赖模型的输入输出关系，不需要打开黑箱，是 LRP 归因图的实用替代品。"),
        Step("⑦ 结论对照",
             "把复现得到的关键数字与原文的三条主张逐条对齐：融合优于单模态、"
             "深度特征提取可建模为连续过程、潜空间轨迹可解释。",
             s7, "metrics",
             "合成数据只支持方向性结论；绝对 AUC 与真实临床场景没有可比性。"),
    ]
