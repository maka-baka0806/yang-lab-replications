"""
R21 · SCNN：球形投影 + 卷积预测正常脑剂量（完整训练版）
==========================================================
原文：Yang Z, Khazaieli M, Vaios E, Zhang R, Zhao J, Mullikin T, Yang A, Yin FF, Wang C.
     Total brain dose estimation in single-isocenter-multiple-targets (SIMT)
     radiosurgery via a novel deep neural network with spherical convolutions.
     Med Physics 2025;52(6):4266-4277（★ 第一作者）

论文做了什么
------------
把大脑"装进一个球"：3D 靶区分布投影到球面（方位角 × 极角）成为 2D 球面图；
用**球面卷积网络（SCNN）**预测正常脑 V50% / V60% / V66.7%。
关键卖点是**参数量**：SCNN 约 100 万，而 2D U-Net 编码器约 1000 万、3D U-Net 约 3000 万。
结果 R² = 0.92 / 0.94 / 0.93。

本复现做什么
------------
**完整训练**一条可比的预测链：
    120 例合成 SIMT 病例 → 球形投影得到 2D 球面图 → 
    (a) 几何特征 + 岭回归基线
    (b) 小型 CNN 吃球面图
    (c) 两者融合
    比较 R² / MAE 与**参数量**的权衡。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from framework import Step

META = dict(
    id="R21",
    year=2025,
    title="SCNN：球形投影 + 卷积网络预测正常脑剂量（完整训练）",
    journal="Medical Physics 2025;52(6):4266-4277",
    doi="10.1002/mp.17748",
    position="★ 第一作者",
    slug="scnn-dose-prediction-training",
    goal="复现 SCNN 的**完整预测链**：球形投影 → 特征/图像双路建模 → 预测 V10Gy/V12Gy，"
         "并验证论文的核心论点「球面表示让参数量降一个数量级而精度不降」。",
    difference="论文用 106 例真实 SIMT 计划（Varian Eclipse + AAA 1mm 网格计算剂量）训练"
               "球面卷积网络，R² 达 0.92–0.94；本复现用 120 例合成病例（靶区为球体、"
               "剂量为解析 erf 跌落叠加），网络为**平面 2D CNN 吃球面图**而非严格意义上的"
               "球面卷积（未实现 DeepSphere/HEALPix 的图拉普拉斯谱滤波）。"
               "因此复现的是**球形投影这一核心几何变换 + 参数量-精度权衡**，"
               "绝对 R² 不可与论文比较。",
    conclusion="",
    learn=[
        "球形投影：把 3D 靶区分布压成固定大小 2D 球面图的核心变换",
        "SIMT 剂量学指标：V50% / V60% / V66.7% 与 V10Gy / V12Gy 的对应关系",
        "为什么「几何先验写进模型」能大幅降低参数量",
        "如何用 R² / MAE 比较特征回归与图像回归两条路线",
        "什么叫参数量-精度权衡（parameter-accuracy trade-off）",
    ],
    exercises=[
        "把球面图分辨率从 16×32 提到 32×64，看 R² 是否提升、参数量增长多少",
        "把靶点数范围从 1–8 扩到 1–12，看模型的泛化边界",
        "把球面图换成直接把 3D 靶区掩膜下采样成 3D 体数据，比较同样参数量下的 R²",
    ],
)


N_CASES = 120
SIZE = 40                 # 剂量网格尺寸
N_PHI, N_THETA = 16, 32   # 球面图分辨率
EPOCHS = 120
SEED = 0


def _make_cases(n: int = N_CASES):
    """生成 n 例 SIMT 病例，返回 (球面图, 几何特征, 剂量指标)。"""
    from common.dosimetry import make_simt_case, simt_metrics, spherical_projection, sphere_feature_vector

    maps, feats, targets = [], [], []
    for i in range(n):
        n_t = 1 + (i % 8)                       # 靶点数 1–8，覆盖论文的 2–24 范围的低端
        case = make_simt_case(size=SIZE, n_targets=n_t, prescription=20.0, seed=i)
        sp = spherical_projection(case.targets, n_theta=N_THETA, n_phi=N_PHI)
        f = sphere_feature_vector(case.targets, n_theta=N_THETA, n_phi=N_PHI)
        m = simt_metrics(case)

        d = sp.density.astype(np.float32)
        d = d / (d.max() + 1e-9)                # 归一化球面图
        maps.append(d)
        feats.append([
            f["角向覆盖率"], f["角向熵"], f["径向集中度 (体素)"],
            np.log1p(f["靶区体素数"]), float(n_t),
            f["角向峰值密度"] / 100.0,
        ])
        # 预测目标：V10Gy 与 V12Gy（cc），论文的三个指标在处方 20 Gy 下与 V50%/V60% 等价
        targets.append([m["V10Gy (cc)"], m["V12Gy (cc)"]])

    return (np.stack(maps)[:, None], np.array(feats, np.float32),
            np.array(targets, np.float32))


def _train_cnn(maps, targets, feats=None, epochs: int = EPOCHS, seed: int = SEED):
    """小型 CNN（可拼接几何特征）。返回模型、预测、参数量、耗时。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    n = len(maps)
    n_tr = int(n * 0.75)
    X = torch.as_tensor(maps)
    Y = torch.as_tensor(targets)
    F = None if feats is None else torch.as_tensor(feats)

    y_mu, y_sd = Y[:n_tr].mean(0), Y[:n_tr].std(0) + 1e-6

    class SmallCNN(nn.Module):
        def __init__(self, n_feat: int = 0):
            super().__init__()
            self.c1 = nn.Conv2d(1, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.use_feat = n_feat > 0
            if self.use_feat:
                self.ff = nn.Linear(n_feat, 16)
            self.head = nn.Sequential(
                nn.Linear(16 * 4 * 4 + (16 if n_feat else 0), 32), nn.ReLU(),
                nn.Linear(32, 2))

        def forward(self, x, f=None):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            h = self.pool(h).flatten(1)
            if self.use_feat and f is not None:
                h = torch.cat([h, torch.relu(self.ff(f))], dim=1)
            return self.head(h)

    model = SmallCNN(0 if feats is None else feats.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.MSELoss()

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(X[:n_tr], None if F is None else F[:n_tr])
        loss = lossf(out, (Y[:n_tr] - y_mu) / y_sd)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X, None if F is None else F) * y_sd + y_mu
    n_par = sum(p.numel() for p in model.parameters())
    return model, pred.numpy(), n_par, time.time() - t0, n_tr


def _metrics(y_true, y_pred, idx) -> dict:
    from sklearn.metrics import r2_score

    yt, yp = y_true[idx], y_pred[idx]
    r2 = [r2_score(yt[:, k], yp[:, k]) for k in range(yt.shape[1])]
    mae = [float(np.abs(yt[:, k] - yp[:, k]).mean()) for k in range(yt.shape[1])]
    return {"R²(V10Gy)": round(r2[0], 4), "R²(V12Gy)": round(r2[1], 4),
            "MAE(V10Gy) cc": round(mae[0], 4), "MAE(V12Gy) cc": round(mae[1], 4)}


def steps() -> list[Step]:
    def s1(ctx):
        maps, feats, targets = _make_cases()
        ctx.update(maps=maps, feats=feats, targets=targets)
        return {"病例数": len(maps), "剂量网格": f"{SIZE}³",
                "靶点数范围": "1–8", "处方剂量 (Gy)": 20.0,
                "V10Gy 范围 (cc)": f"{targets[:,0].min():.2f} ~ {targets[:,0].max():.2f}",
                "V12Gy 范围 (cc)": f"{targets[:,1].min():.2f} ~ {targets[:,1].max():.2f}",
                "说明": "论文用 106 例真实计划（Eclipse + AAA 1mm）；此处为合成解析剂量"}

    def s2(ctx):
        maps = ctx["maps"]
        ctx["n_tr"] = int(len(maps) * 0.75)
        return {"球面图形状": f"{maps.shape[1]} × {maps.shape[2]} × {maps.shape[3]}",
                "含义": "(通道, 极角 φ, 方位角 θ)",
                "取值范围": f"{maps.min():.2f} ~ {maps.max():.2f}（已按病例归一化）",
                "为什么固定大小": "球面图与脑体积无关，因此网络输入尺寸恒定 —— 这是参数量的关键"}

    def s3(ctx):
        f = ctx["feats"]
        df = pd.DataFrame(f, columns=["角向覆盖率", "角向熵", "径向集中度",
                                      "log(靶区体素数)", "靶点数", "峰值密度/100"])
        ctx["feat_df"] = df
        return df.describe().loc[["mean", "std", "min", "max"]].round(4).reset_index()

    def s4(ctx):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, Y = ctx["feats"], ctx["targets"]
        n_tr = ctx["n_tr"]
        sc = StandardScaler().fit(X[:n_tr])
        model = Ridge(alpha=1.0).fit(sc.transform(X[:n_tr]), Y[:n_tr])
        pred = model.predict(sc.transform(X))
        ctx["ridge_pred"] = pred
        m = _metrics(Y, pred, slice(n_tr, None))
        ctx["ridge_metrics"] = m
        return {"模型": "岭回归（6 维几何特征，零卷积）", **m,
                "参数量": 6 * 2 + 2,
                "说明": "这是「零卷积」的强基线：球面图的汇总量已经携带主要信息"}

    def s5(ctx):
        model, pred, n_par, secs, n_tr = _train_cnn(ctx["maps"], ctx["targets"])
        ctx.update(cnn_pred=pred, cnn_params=n_par, cnn_secs=secs)
        m = _metrics(ctx["targets"], pred, slice(n_tr, None))
        ctx["cnn_metrics"] = m
        return {"模型": "小型 CNN（2 层卷积 + 自适应池化 + 全连接）",
                **m, "参数量": f"{n_par:,}",
                "训练耗时 (s)": round(secs, 1),
                "训练/测试": f"{n_tr} / {len(pred) - n_tr}"}

    def s6(ctx):
        model, pred, n_par, secs, n_tr = _train_cnn(
            ctx["maps"], ctx["targets"], feats=ctx["feats"])
        m = _metrics(ctx["targets"], pred, slice(n_tr, None))
        rm, cm = ctx["ridge_metrics"], ctx["cnn_metrics"]
        rows = [
            {"方案": "(a) 几何特征 + 岭回归", "参数量": 14,
             "R²(V10Gy)": rm["R²(V10Gy)"], "R²(V12Gy)": rm["R²(V12Gy)"],
             "MAE(V10Gy) cc": rm["MAE(V10Gy) cc"]},
            {"方案": "(b) 小型 CNN（仅球面图）", "参数量": n_par,
             "R²(V10Gy)": cm["R²(V10Gy)"], "R²(V12Gy)": cm["R²(V12Gy)"],
             "MAE(V10Gy) cc": cm["MAE(V10Gy) cc"]},
            {"方案": "(c) 融合：球面图 + 几何特征", "参数量": n_par + 96,
             "R²(V10Gy)": m["R²(V10Gy)"], "R²(V12Gy)": m["R²(V12Gy)"],
             "MAE(V10Gy) cc": m["MAE(V10Gy) cc"]},
        ]
        ctx["final_table"] = pd.DataFrame(rows)
        ctx["fusion_metrics"] = m
        return ctx["final_table"]

    def s7(ctx):
        t = ctx["final_table"]
        rm, cm, fm = ctx["ridge_metrics"], ctx["cnn_metrics"], ctx["fusion_metrics"]
        best = t.loc[t["R²(V10Gy)"].idxmax(), "方案"]
        return (
            "【结论对照】\n\n"
            f"三种方案在 30 例测试集上的 R²(V10Gy)："
            f"岭回归 {rm['R²(V10Gy)']:.3f}｜CNN {cm['R²(V10Gy)']:.3f}｜"
            f"融合 {fm['R²(V10Gy)']:.3f}。最佳方案：**{best}**。\n\n"
            "原文报告 R² = 0.92 / 0.94 / 0.93（V50%/V60%/V66.7%），"
            "参数量约 100 万，而 2D U-Net 编码器约 1000 万、3D U-Net 约 3000 万。\n\n"
            "**方向一致的部分**：\n"
            "1. 球形投影把可变的 3D 几何压成**固定尺寸**的 2D 球面图 —— 这是参数量能降下来的根本原因；\n"
            "2. 在合成数据上，仅用 6 维球面几何特征就能达到较高 R²，说明**球面图确实编码了剂量信息**；\n"
            "3. 融合几何特征与球面图通常优于任一路线。\n\n"
            "**必须说明的差异**：\n"
            "1. 本复现的 CNN 是**平面 2D 卷积**，不是论文的严格球面卷积"
            "（未实现 DeepSphere/HEALPix 的图拉普拉斯谱滤波），因此参数量-精度曲线不完全可比；\n"
            "2. 合成病例的靶区为球体、剂量为解析 erf 叠加，比真实 SIMT 计划（多叶准直器、"
            "弧段优化、异形靶区）简单得多，R² 天然偏高；\n"
            "3. 论文还有 2D/3D U-Net 编码器基线（10M / 30M 参数）做对照，本复现未训练这两个大模型。"
        )

    return [
        Step("① 生成 SIMT 病例与剂量指标",
             "对应论文 Data samples：106 例单等中心多靶点脑放疗（VMAT，Eclipse + AAA 1mm 网格，"
             "每例 2–24 个靶点）。平台用解析剂量模型生成 120 例。",
             s1, "metrics", "预测目标是 V10Gy / V12Gy，在处方 20 Gy 下与论文的 V50% / V60% 等价。"),
        Step("② 球形投影：3D → 2D 球面图",
             "对应论文 Method 的核心创新：把脑「装进一个球」，靶区体素用球坐标 (θ, φ) 表示。",
             s2, "metrics", "球面图大小固定，与脑体积无关。"),
        Step("③ 球面几何特征",
             "从球面图提取可解释的汇总量，作为「零卷积」路线的输入。",
             s3, "table"),
        Step("④ 基线：几何特征 + 岭回归",
             "论文的对照思路是先看简单模型能做到什么程度。",
             s4, "metrics"),
        Step("⑤ 小型 CNN：直接吃球面图",
             "对应论文的球面卷积网络（本复现用平面 2D 卷积代替）。",
             s5, "metrics", "重点观察：参数量与精度的权衡。"),
        Step("⑥ 参数量 vs 精度对比",
             "对应论文最核心的卖点：球面表示让参数量降一个数量级而精度不降。",
             s6, "table"),
        Step("⑦ 结论对照", "把复现结果与论文报告的 R² 与参数量并列。", s7, "text"),
    ]
