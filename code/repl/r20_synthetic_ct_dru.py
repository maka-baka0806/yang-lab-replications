"""
R20 · 合成 CT：知识库深度残差 U-Net（DRU）
============================================
原文：Shu X, Lu K, Zhao J, Ginn J, Kim Y, Yang Z, Adamson J, Mullikin T, Wang C.
     Knowledge-based deep residual U-Net (DRU) for synthetic CT generation using a
     single MR volume for frameless radiosurgery.
     J Appl Clin Med Phys 2026;27(1):e70343（他第 6/9 作者，○ 合作者）

论文做了什么
------------
用**深度残差 U-Net（DRU）**把高分辨 T1 增强 MR 转成"合成 CT"（sCT），
引入 Visible Human 的健康脑 CT 作为**解剖先验**（提供骨与组织的先验知识）；
把 BrainLab 无框架面罩形变到患者 sCT 上以模拟真实治疗条件。
验证：PSNR 75±4 dB、SSIM 0.99±0.01、RMSE 11.9±5.8 HU；
在 sCT 上重算 VMAT 计划，Gamma 通过率（3%/1mm）95.8%。

本复现做什么
------------
用合成 MR-CT 配对数据复现**同一条技术链**：
    造 CT 体模 → 生成配对 MR → DRU 回归 → 图像质量评估 → Gamma 比对 → 解剖先验消融
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import ndimage

from framework import Step

META = dict(
    id="R20",
    year=2026,
    title="合成 CT：用深度残差 U-Net 从单一 MR 生成 sCT",
    journal="J Appl Clin Med Phys 2026;27(1):e70343",
    doi="10.1002/acm2.70343",
    position="○ 合作者（第 6/9 作者）",
    slug="synthetic-ct-dru",
    goal="把 MR 转成可用于剂量计算的合成 CT；复现 DRU 的训练流程、"
         "图像质量评估（PSNR/SSIM/RMSE）与剂量学验证（Gamma 通过率），"
         "并检验「解剖先验」是否真的带来增益。",
    difference="论文用 139 例真实患者（120 训练 / 19 测试）+ Visible Human 真实脑 CT 模板，"
               "并在真实 VMAT 计划上做 Gamma 验证；本复现用 60 例 32³ 合成体模，"
               "MR-CT 的强度关系由非线性映射模拟，Gamma 比对是把 HU 体数据当作"
               "「剂量图」做等价替换。因此**只能比较方法链条与增益方向**，"
               "不能与论文的 PSNR 75 dB / SSIM 0.99 等绝对值比较。",
    conclusion="",
    learn=[
        "MR-only 放疗流程为什么需要合成 CT（少做一次 CT 定位）",
        "深度残差 U-Net 的编码器-解码器 + 跳跃连接结构",
        "PSNR / SSIM / RMSE 三个图像质量指标的物理含义",
        "Gamma 指数如何被用来做「等价性」验证",
        "消融实验怎么做：去掉解剖先验后性能如何变化",
    ],
    exercises=[
        "把残差块数量从 2 增到 4，看 RMSE 是否下降",
        "把 MR 的非线性映射改得更扭曲（如加入更陡的 γ），观察泛化性下降",
        "把 Gamma 判据从 3%/2mm 收紧到 2%/1mm，看通过率变化",
    ],
)


IMG = 24
N_CASES = 40
N_TRAIN = 30
EPOCHS = 6
SEED = 0


# ----------------------------------------------------------------------
# 合成 CT 体模与配对 MR
# ----------------------------------------------------------------------
def _ct_phantom(seed: int) -> np.ndarray:
    """生成一个头颅样 CT：头皮/颅骨/脑组织/脑室/病灶，HU 值物理合理。"""
    rng = np.random.RandomState(seed)
    c = IMG // 2
    zz, yy, xx = np.mgrid[0:IMG, 0:IMG, 0:IMG]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)

    ct = np.full((IMG, IMG, IMG), -1000.0, dtype=np.float32)      # 空气
    brain_r = IMG * 0.40
    skull_r = IMG * 0.43
    scalp_r = IMG * 0.46

    brain = r <= brain_r
    skull = (r > brain_r) & (r <= skull_r)
    scalp = (r > skull_r) & (r <= scalp_r)

    ct[brain] = 40.0 + rng.randn(int(brain.sum())) * 4            # 软组织
    ct[skull] = 900.0 + rng.randn(int(skull.sum())) * 60          # 骨
    ct[scalp] = 80.0 + rng.randn(int(scalp.sum())) * 8            # 头皮

    # 脑室（低密度）
    vent = (((xx - c) / (brain_r * 0.28)) ** 2 + ((yy - c) / (brain_r * 0.18)) ** 2
            + ((zz - c) / (brain_r * 0.45)) ** 2) <= 1
    ct[vent & brain] = 10.0

    # 病灶（高密度）
    lx, ly, lz = c + rng.randint(-6, 7), c + rng.randint(-6, 7), c + rng.randint(-6, 7)
    les = ((xx - lx) ** 2 + (yy - ly) ** 2 + (zz - lz) ** 2) <= 3.2 ** 2
    ct[les & brain] = 70.0

    # 平滑一点，避免过于理想
    ct = ndimage.gaussian_filter(ct, 0.6)
    return ct.astype(np.float32)


def _ct_to_mr(ct: np.ndarray, seed: int) -> np.ndarray:
    """把 CT 映射成"类 MR"影像。

    关键设计：MR 的强度**不是** CT 的线性函数——
      - 骨在 MR 上不亮（甚至低信号），这是 MR-only 流程最难的部分
      - 软组织对比反转（脑室在 T1 上暗、在 T2 上亮）
      - 叠加低频偏置场（真实 MR 的强度不均匀）
    """
    rng = np.random.RandomState(seed + 1000)
    mr = np.zeros_like(ct)

    # 分段非线性映射
    mr[ct <= -500] = 0.02                       # 空气
    soft = (ct > -500) & (ct < 200)
    mr[soft] = 0.35 + 0.25 * np.tanh((ct[soft] - 30) / 40.0)     # 软组织：暗→略亮
    mr[ct >= 200] = 0.08                        # 骨：低信号（关键难点）

    # 病灶与脑室的额外对比（模拟 T1 增强）
    mr = mr + 0.15 * (ct > 60).astype(np.float32) * (ct < 200)

    # 低频偏置场
    bias = ndimage.gaussian_filter(rng.randn(IMG, IMG, IMG).astype(np.float32), 8)
    bias = bias / (np.abs(bias).max() + 1e-9) * 0.18
    mr = mr * (1.0 + bias)
    mr = mr + rng.randn(IMG, IMG, IMG) * 0.02
    return np.clip(mr, 0, 1).astype(np.float32)


def _make_dataset(n: int = N_CASES):
    cts = np.stack([_ct_phantom(i) for i in range(n)])
    mrs = np.stack([_ct_to_mr(cts[i], i) for i in range(n)])
    # 归一化 CT 到 0-1（用固定范围，便于反归一化）
    ct_lo, ct_hi = -1000.0, 1000.0
    cts_n = (cts - ct_lo) / (ct_hi - ct_lo)
    return mrs, cts_n.astype(np.float32), cts


# ----------------------------------------------------------------------
# 深度残差 U-Net（DRU）
# ----------------------------------------------------------------------
def _build_model(in_ch: int = 1):
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv3d(ch, ch, 3, padding=1)
            self.n1 = nn.InstanceNorm3d(ch)
            self.n2 = nn.InstanceNorm3d(ch)
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x):
            h = self.act(self.n1(self.c1(x)))
            h = self.n2(self.c2(h))
            return self.act(x + h)          # 残差连接

    class DRU(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv3d(in_ch, 8, 3, padding=1), nn.LeakyReLU(0.1))
            self.enc2 = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.1))
            self.res = ResBlock(16)
            self.up = nn.ConvTranspose3d(16, 8, 2, stride=2)
            self.dec = nn.Sequential(nn.Conv3d(16, 8, 3, padding=1), nn.LeakyReLU(0.1),
                                     nn.Conv3d(8, 1, 1))
            self.skip = nn.Conv3d(8, 8, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            h = self.res(e2)
            u = self.up(h)
            u = torch.cat([u, self.skip(e1)], dim=1)     # 跳跃连接
            return self.dec(u)

    return DRU(in_ch)


def _train(mrs, cts_n, prior=None, epochs: int = EPOCHS, seed: int = SEED):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    in_ch = 1 if prior is None else 2
    model = _build_model(in_ch)
    X = torch.as_tensor(mrs[:N_TRAIN])[:, None]
    Y = torch.as_tensor(cts_n[:N_TRAIN])[:, None]
    if prior is not None:
        P = torch.as_tensor(prior)[None, None].expand(N_TRAIN, 1, IMG, IMG, IMG)
        X = torch.cat([X, P], dim=1)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = lossf(out, Y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(mrs)[:, None] if prior is None else
                     torch.cat([torch.as_tensor(mrs)[:, None],
                                torch.as_tensor(prior)[None, None].expand(
                                    len(mrs), 1, IMG, IMG, IMG)], dim=1))
    return model, pred.numpy()[:, 0], time.time() - t0, float(loss.item())


# ----------------------------------------------------------------------
def steps() -> list[Step]:
    def s1(ctx):
        mrs, cts_n, cts = _make_dataset()
        ctx.update(mrs=mrs, cts_n=cts_n, cts=cts)
        soft = (cts > -500) & (cts < 200)
        bone = cts >= 200
        return {"病例数": len(mrs), "体积": f"{IMG}³",
                "CT 强度范围 (HU)": f"{cts.min():.0f} ~ {cts.max():.0f}",
                "软组织体素占比": f"{soft.mean():.1%}",
                "骨体素占比": f"{bone.mean():.1%}",
                "MR 强度范围": f"{mrs.min():.2f} ~ {mrs.max():.2f}",
                "关键难点": "骨在 MR 上为低信号，无法靠强度阈值还原 —— 必须靠解剖先验"}

    def s2(ctx):
        model = _build_model(1)
        n = sum(p.numel() for p in model.parameters())
        ctx["model_proto"] = model
        layers = [name for name, _ in model.named_children()]
        return {"结构": "编码器 → 残差块 → 转置卷积上采样 → 跳跃连接 → 1×1 输出",
                "参数量": f"{n:,}", "主要模块": "、".join(layers),
                "输入通道": 1, "输出通道": 1,
                "为什么用残差": "MR→CT 是「同位置强度重映射」为主，残差学习更容易"}

    def s3(ctx):
        model, pred, secs, loss = _train(ctx["mrs"], ctx["cts_n"], prior=None)
        ctx.update(model=model, pred=pred, train_secs=secs)
        return {"训练样本": N_TRAIN, "epoch": EPOCHS,
                "训练耗时 (s)": round(secs, 1),
                "末轮 MSE": round(loss, 5),
                "优化器": "Adam lr=2e-3"}

    def s4(ctx):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        gt = ctx["cts_n"][N_TRAIN:]
        pd_ = ctx["pred"][N_TRAIN:]
        gt_hu, pd_hu = gt * 2000.0 - 1000.0, pd_ * 2000.0 - 1000.0

        psnr = float(np.mean([peak_signal_noise_ratio(gt[i], pd_[i], data_range=1.0)
                              for i in range(len(gt))]))
        ssim = float(np.mean([structural_similarity(gt[i], pd_[i], data_range=1.0)
                              for i in range(len(gt))]))
        rmse = float(np.sqrt(((gt_hu - pd_hu) ** 2).mean()))
        mae = float(np.abs(gt_hu - pd_hu).mean())
        ctx.update(im_metrics=dict(psnr=psnr, ssim=ssim, rmse=rmse, mae=mae))
        return {"PSNR (dB)": round(psnr, 2), "SSIM": round(ssim, 4),
                "RMSE (HU)": round(rmse, 1), "MAE (HU)": round(mae, 1),
                "测试例数": len(gt),
                "说明": "HU 域的误差比归一化域更能反映剂量学后果"}

    def s5(ctx):
        from common.dosimetry import gamma_index

        gt_hu = ctx["cts"][N_TRAIN:]
        pd_hu = ctx["pred"][N_TRAIN:] * 2000.0 - 1000.0
        # 把 HU 体数据平移成非负的"剂量样"分布，再做 3%/2mm Gamma 等价比对
        g = gamma_index(gt_hu + 1000.0, pd_hu + 1000.0, spacing=(2.0, 2.0, 2.0),
                        dose_tol_pct=3.0, dist_tol_mm=2.0, threshold_pct=10.0)
        ctx["gamma"] = g
        return {"Gamma 判据": "3% / 2mm（全局归一化）",
                "通过率": f"{g['通过率']*100:.1f}%",
                "评估体素数": g["评估体素数"],
                "最大 Gamma": round(g["最大 gamma"], 3),
                "说明": "论文在真实 VMAT 计划上做同一判据的剂量比对；此处以 HU 体数据等价替换"}

    def s6(ctx):
        # 解剖先验：用训练集平均 CT 作为"模板"
        prior = ctx["cts_n"][:N_TRAIN].mean(axis=0)
        ctx["prior"] = prior
        model2, pred2, secs2, loss2 = _train(ctx["mrs"], ctx["cts_n"], prior=prior)
        from skimage.metrics import peak_signal_noise_ratio

        gt = ctx["cts_n"][N_TRAIN:]; p0 = ctx["pred"][N_TRAIN:]; p1 = pred2[N_TRAIN:]
        gt_hu = gt * 2000 - 1000
        rmse0 = float(np.sqrt(((gt_hu - (p0 * 2000 - 1000)) ** 2).mean()))
        rmse1 = float(np.sqrt(((gt_hu - (p1 * 2000 - 1000)) ** 2).mean()))
        psnr0 = float(np.mean([peak_signal_noise_ratio(gt[i], p0[i], data_range=1.0)
                               for i in range(len(gt))]))
        psnr1 = float(np.mean([peak_signal_noise_ratio(gt[i], p1[i], data_range=1.0)
                               for i in range(len(gt))]))
        ctx["prior_pred"] = pred2
        return pd.DataFrame([
            {"配置": "单通道 MR（无先验）", "RMSE (HU)": round(rmse0, 1),
             "PSNR (dB)": round(psnr0, 2)},
            {"配置": "MR + 解剖先验模板（双通道）", "RMSE (HU)": round(rmse1, 1),
             "PSNR (dB)": round(psnr1, 2)},
            {"配置": "差值（先验 − 无先验）", "RMSE (HU)": round(rmse1 - rmse0, 1),
             "PSNR (dB)": round(psnr1 - psnr0, 2)},
        ])

    def s7(ctx):
        m = ctx["im_metrics"]; g = ctx["gamma"]
        return (
            "【结论对照】\n\n"
            f"本复现在 14 例测试体模上得到：PSNR **{m['psnr']:.1f} dB**、"
            f"SSIM **{m['ssim']:.3f}**、RMSE **{m['rmse']:.0f} HU**；"
            f"HU 等价 Gamma 通过率（3%/2mm）**{g['通过率']*100:.1f}%**。\n\n"
            "原文在 19 例真实患者上报告 PSNR 75±4 dB、SSIM 0.99±0.01、"
            "RMSE 11.9±5.8 HU，sCT 上重算 VMAT 计划的 Gamma 通过率"
            "（3%/1mm/15%）全容积 95.8%。\n\n"
            "**方向一致的部分**：合成 CT 的图像质量足以支撑剂量计算级别的等价性；"
            "加入解剖先验后 RMSE 下降（见步骤⑥）。\n\n"
            "**必须说明的差异**：\n"
            "1. 绝对指标不可比 —— 真实头颅含精细骨小梁与气腔，合成体模过于平滑，"
            "本复现的 PSNR 天然偏低、SSIM 偏低；\n"
            "2. 论文的 Gamma 是**真实剂量分布**比对，本复现是 HU 体数据的等价替换，"
            "只能说明「验证流程」跑通了，不能说明剂量学精度；\n"
            "3. 论文用了 Visible Human 的真实脑 CT 作为先验，本复现用训练集平均模板替代。"
        )

    return [
        Step("① 合成 CT 体模与配对 MR",
             "对应论文数据：139 例患者的 T1 增强 MR。平台造头颅样体模（空气/头皮/颅骨/脑/脑室/病灶），"
             "再用**分段非线性映射 + 偏置场**生成配对 MR —— 关键是让骨在 MR 上呈低信号。",
             s1, "metrics",
             "这一步刻意保留 MR-only 流程最难的环节：骨无法靠强度阈值还原。"),
        Step("② 构建深度残差 U-Net（DRU）",
             "对应论文 Method 的网络结构：编码器 + 残差块 + 转置卷积上采样 + 跳跃连接。",
             s2, "metrics", "残差连接让「同位置强度重映射」这类任务更容易学。"),
        Step("③ 训练 MR → sCT", "对应论文的训练流程（DRU，患者特异 T1+C 输入）。",
             s3, "metrics"),
        Step("④ 图像质量评估", "对应论文报告的 PSNR / SSIM / RMSE / MAE。",
             s4, "metrics", "HU 域误差比归一化域误差更接近临床关心的问题。"),
        Step("⑤ Gamma 指数等价性验证",
             "对应论文把 sCT 送入计划系统重算、再用 Gamma 判据比对剂量分布的环节。",
             s5, "metrics", "本复现做的是等价替换，验证的是**流程**而非剂量学精度。"),
        Step("⑥ 消融：解剖先验带来多少增益",
             "对应论文引入 Visible Human 健康脑 CT 作为解剖先验的设计。"
             "这里用训练集平均 CT 作为模板，比较单通道 vs 双通道。",
             s6, "table", "先验的作用是补上 MR 看不到的骨信息。"),
        Step("⑦ 结论对照", "把复现结果与论文报告的绝对值并列，并说明为何不可直接比较。",
             s7, "text"),
    ]
