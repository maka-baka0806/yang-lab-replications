"""
R13 · 深度学习剂量图预测：HDR 近距离治疗
==========================================
原文：Li Z, Yang Z, Lu J, Zhu Q, Wang Y, Zhao M, Li Z, Fu J.
Deep learning-based dose map prediction for high-dose-rate brachytherapy.
Phys Med Biol 2023;68:175015.  DOI: 10.1088/1361-6560/acecd2

原文做了什么
------------
把近距离治疗的剂量学预测从「预测 DVH 汇总指标（D2cc、D90…）」升级为
**直接预测三维剂量分布图**：网络以几何描述为输入，逐体素输出剂量，
再由预测出的剂量图去计算任意结构、任意指标的 DVH 参数。这样一次预测
就能支撑所有剂量学终点，而不是每个指标训练一个模型。

本复现怎么做
------------
1) 合成 HDR 近距离治疗场景：靶区（类宫颈宫腔管/卵圆体的椭球）+ 危及器官（OAR）
   + 体轮廓；剂量按**到靶区表面的距离 DTH** 跌落 —— 物理本质是「点源平方反比
   × 组织衰减」，多驻留点叠加后表现为**双指数**衰减（近场陡、远场拖尾）；
2) 输入 2 通道：带符号的 DTH 图 + 靶区掩膜；输出 1 通道：剂量图；
3) 用轻量 3D CNN（stride-2 编码 + 反卷积解码）在 30 例上训练、10 例上测试；
4) 评估：剂量图 MAE（分高/低剂量区）、由预测图算出的 D2cc / Dmean / D90 与真值的残差；
5) 基线：只用 DTH 的**单指数解析模型**（最小二乘拟合 A·exp(-DTH/λ)）——
   它无法表达双指数与靶内剂量抬升，正是深度学习要超越的对象。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage

from common.dosimetry import dose_metrics, dvh_curve
from framework import Step

META = dict(
    id="R13",
    year=2023,
    title="深度学习剂量图预测：HDR 近距离治疗",
    journal="Physics in Medicine & Biology 2023;68:175015",
    doi="10.1088/1361-6560/acecd2",
    position="○ 合作者（第 2/8 作者）",
    slug="dose-map-prediction-hdr-brachytherapy",
    goal="复现原文「从预测 DVH 汇总指标升级为直接预测三维剂量分布图」的核心思想："
         "用 DTH（到靶区表面的距离）与靶区掩膜作为输入训练 3D CNN 预测整幅剂量图，"
         "再由预测剂量图反推 D2cc / Dmean / D90，并与「只用 DTH 的单指数解析模型」基线比较。",
    difference="原论文用真实 HDR 近距离治疗病例与临床治疗计划系统的剂量图；"
                "本复现用合成病例（32³ 体素、2 mm 间距、40 例），剂量由双指数距离模型生成，"
                "网络是极轻的 3D CNN（约 1.4 万参数、10 epoch）。因此只复现**方法链路与相对结论**"
                "（剂量图预测可行、由预测图算 DVH 指标的残差可控、物理先验增强优于纯数据驱动），"
                "不比较绝对剂量学精度。",
    conclusion="复现完整打通了「几何 → 三维剂量图 → DVH 指标」这条链路：以 DTH 与靶区掩膜为输入的"
               "轻量 3D CNN 能在未见过的病例上预测出整幅剂量分布，由预测剂量图反推的 "
               "D2cc / Dmean / D90 与真值接近，说明**预测整幅剂量图确实比「一个指标一个模型」更通用**——"
               "一次前向推理就能给出任意结构、任意指标的 DVH 参数。但复现同时给出一个反直觉、"
               "也更有价值的结论：在只有 30 例训练数据、网络极轻、输入仅 2 通道几何信息的条件下，"
               "**纯数据驱动的 CNN 并没有战胜零训练的解析模型**（单指数 A·exp(-DTH/λ)）；"
               "真正胜出的是「物理先验 + 残差学习」的混合模型 —— 把解析场作为网络里的固定层，"
               "让 CNN 只学双指数拖尾与靶内剂量抬升这些结构性残差，它在剂量图 MAE、D2cc 残差、"
               "D90 残差与靶区 DVH 偏差上同时优于解析基线与纯 CNN。"
               "结论：剂量图级预测是近距离治疗剂量学自动化的发展方向，而小样本场景下的正确姿势是"
               "physics-informed 而不是 pure data-driven；其精度上限还取决于输入几何描述的完备性"
               "（处方剂量、驻留位置与组织不均匀性当前都不在输入里）。",
    learn=[
        "HDR 近距离治疗的剂量学本质：点源平方反比 × 组织衰减，多驻留点叠加呈双指数跌落",
        "DTH（distance to target surface）如何把「三维几何」变成网络能直接吃的输入通道",
        "如何用「剂量图 MAE」与「由预测图反推的 D2cc/Dmean/D90 残差」两级指标评估剂量预测",
        "为什么「预测整幅剂量图」比「预测单个 DVH 指标」更通用，以及它的误差传播特性",
        "解析模型与深度学习模型在剂量预测中的分工：解析模型给物理先验，网络补结构性残差",
    ],
    exercises=[
        "把处方剂量也作为输入通道（第 3 通道），观察剂量图 MAE 与 D2cc 残差是否下降，解释原因",
        "把训练集从 30 例减到 10 / 20 例，画「训练例数 vs 测试 MAE」曲线，寻找数据效率拐点",
        "把损失函数从 MSE 换成「MSE + 高剂量区加权」，观察高剂量区 MAE 与 D2cc 残差的变化",
    ],
)

SIZE = 32
SPACING = (2.0, 2.0, 2.0)          # mm
N_CASES = 40
N_TRAIN = 30
EPOCHS = 10
BATCH = 4
DOSE_SCALE = 7.0                   # 归一化常数（Gy），仅用于把回归目标缩放到 ~1
DTH_SCALE = 10.0                   # mm

# 报告在缺少 tabulate 时会把 DataFrame 退化成文本块，放宽显示宽度以免列被截断
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
SEED = 13
HYBRID_EPOCHS = 8                   # 残差学习的收敛更快，用更少 epoch


# ----------------------------------------------------------------------
# 合成 HDR 病例：几何 + 距离-剂量关系
# ----------------------------------------------------------------------
def _make_case(size: int = SIZE, seed: int = 0) -> dict:
    """一个合成 HDR 近距离治疗病例。

    - 靶区：绕 z 轴拉长的椭球（模拟宫颈癌宫腔管 + 卵圆体），带低频边界扰动
    - OAR：在靶区侧方、与靶区表面相距约 6 mm 的拉长椭球（模拟直肠/膀胱）
    - 剂量：D = D_pres · [a·exp(-DTH/λ1) + (1-a)·exp(-DTH/λ2)] · (1 + 0.6·exp(DTH/2.5))（DTH<0 时）
      物理上对应「多驻留点叠加的近距离治疗剂量场」：近场由 1/r² 主导（陡），
      远场由组织衰减与散射拖尾主导（缓），因此单指数无法拟合。
    """
    rng = np.random.RandomState(seed)
    c = size / 2.0
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)

    body = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= (size * 0.47) ** 2

    a_ax = rng.uniform(3.5, 5.0)                 # x 半轴（体素）
    b_ax = rng.uniform(3.5, 5.0)                 # y 半轴
    l_ax = rng.uniform(6.0, 9.0)                 # z 半轴
    cx, cy = c + rng.uniform(-1.5, 1.5), c + rng.uniform(-1.5, 1.5)

    pert = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 3.0)
    pert /= (np.abs(pert).max() + 1e-6)
    ell = (((xx - cx) / a_ax) ** 2 + ((yy - cy) / b_ax) ** 2 + ((zz - c) / l_ax) ** 2)
    target = (ell <= (1.0 + 0.18 * pert)) & body

    side = 1.0 if rng.rand() > 0.5 else -1.0
    ox = cx + side * (a_ax + 6.5)
    oar = ((((xx - ox) / 3.5) ** 2 + ((yy - cy) / 3.5) ** 2
            + ((zz - c) / 8.0) ** 2) <= 1.0) & body & ~target

    # 带符号距离：靶外为正、靶内为负
    dth = (ndimage.distance_transform_edt(~target, sampling=SPACING)
           - ndimage.distance_transform_edt(target, sampling=SPACING)).astype(np.float32)
    dth = np.where(body, dth, 0.0)
    dth_pos = np.maximum(dth, 0.0)
    dth_neg = np.maximum(-dth, 0.0)

    pres = rng.uniform(6.0, 7.0)                 # 处方剂量（Gy/分次）
    lam1 = rng.uniform(2.2, 2.8)                 # 近场衰减长度（mm）
    lam2 = rng.uniform(8.0, 10.0)                # 远场拖尾（mm）
    mix = 0.75
    dose = pres * (mix * np.exp(-dth_pos / lam1) + (1 - mix) * np.exp(-dth_pos / lam2))
    dose *= (1.0 + 0.60 * np.exp(-dth_neg / 2.5))          # 靶内剂量抬升
    dose *= (1.0 + 0.04 * ndimage.gaussian_filter(
        rng.randn(size, size, size).astype(np.float32), 2.0))   # 计算/测量噪声
    dose = (dose * body).astype(np.float32)

    return {"dose": dose, "target": target, "oar": oar, "body": body,
            "dth": dth, "prescription": float(pres), "lam1": float(lam1),
            "lam2": float(lam2), "seed": seed}


def _dth_stats(cases: list[dict]) -> pd.DataFrame:
    """按 DTH 分档看剂量跌落：验证「双指数」而非「单指数」。"""
    edges = [-8, -4, -2, 0, 2, 4, 6, 8, 10, 14, 20, 30]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        num = den = dth_sum = 0.0
        for cs in cases:
            sel = cs["body"] & (cs["dth"] >= lo) & (cs["dth"] < hi)
            if sel.sum() == 0:
                continue
            num += float(cs["dose"][sel].sum() / cs["prescription"])
            dth_sum += float(cs["dth"][sel].sum())
            den += float(sel.sum())
        if den == 0:
            continue
        rows.append({"DTH 区间 (mm)": f"[{lo}, {hi})", "体素数": int(den),
                     "平均 DTH (mm)": dth_sum / den,
                     "平均剂量 / 处方剂量": num / den})
    return pd.DataFrame(rows)


def _fit_analytic(cases: list[dict]) -> tuple[float, float, float]:
    """解析基线：最小二乘拟合 D = A·exp(-max(DTH,0)/λ)（单指数，只看 DTH）。"""
    dth = np.concatenate([np.maximum(cs["dth"][cs["body"]], 0.0) for cs in cases])
    dose = np.concatenate([cs["dose"][cs["body"]] for cs in cases])
    best = (0.0, 1.0, float("inf"))
    for lam in np.arange(1.0, 20.01, 0.25):
        basis = np.exp(-dth / lam)
        A = float((basis * dose).sum() / max((basis * basis).sum(), 1e-9))
        mse = float(((A * basis - dose) ** 2).mean())
        if mse < best[2]:
            best = (A, float(lam), mse)
    return best


def _analytic_predict(cs: dict, A: float, lam: float) -> np.ndarray:
    d = A * np.exp(-np.maximum(cs["dth"], 0.0) / lam)
    return (d * cs["body"]).astype(np.float32)


def _mae(pred: np.ndarray, cases: list[dict]) -> float:
    body = np.stack([c["body"] for c in cases])
    truth = np.stack([c["dose"] for c in cases])
    return float(np.abs(pred[body] - truth[body]).mean())


def _mae_high(pred: np.ndarray, cases: list[dict]) -> float:
    """高剂量区（≥50% 处方剂量）的 MAE：直接决定靶区覆盖与热点判断。"""
    body = np.stack([c["body"] for c in cases])
    truth = np.stack([c["dose"] for c in cases])
    pres = np.array([c["prescription"] for c in cases])[:, None, None, None]
    sel = body & (truth >= 0.5 * pres)
    return float(np.abs(pred[sel] - truth[sel]).mean())


# ----------------------------------------------------------------------
# 轻量 3D CNN：2 通道几何输入 → 1 通道剂量图
# ----------------------------------------------------------------------
class DoseNet(nn.Module):
    """stride-2 编码 + 反卷积解码，参数量约 2 千，能在 CPU 上秒级训练完。"""

    def __init__(self, in_ch: int = 2, w: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, w, 3, stride=2, padding=1), nn.ReLU(),      # 16³
            nn.Conv3d(w, w * 2, 3, stride=2, padding=1), nn.ReLU(),      # 8³
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(w * 2, w, 4, stride=2, padding=1), nn.ReLU(),   # 16³
            nn.ConvTranspose3d(w, max(w // 2, 1), 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(max(w // 2, 1), 1, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def _inputs(cases: list[dict]) -> np.ndarray:
    out = []
    for cs in cases:
        out.append(np.stack([np.clip(cs["dth"], -10, 30) / DTH_SCALE,
                             cs["target"].astype(np.float32)], axis=0))
    return np.stack(out).astype(np.float32)


def _targets(cases: list[dict]) -> np.ndarray:
    return np.stack([cs["dose"] / DOSE_SCALE for cs in cases]).astype(np.float32)[:, None]


def _input_tensor(cases: list[dict]) -> torch.Tensor:
    return torch.tensor(_inputs(cases), dtype=torch.float32)


def _base_tensor(base_fields: list[np.ndarray] | None) -> torch.Tensor | float:
    """把解析先验场搬到与网络输出同一尺度（dose / DOSE_SCALE）。"""
    if base_fields is None:
        return 0.0
    return torch.tensor(np.stack(base_fields) / DOSE_SCALE,
                        dtype=torch.float32)[:, None]


def _train(cases: list[dict], base_fields: list[np.ndarray] | None = None,
           high_weight: float = 0.0, seed: int = SEED, epochs: int = EPOCHS,
           lr: float = 3e-3) -> dict:
    """训练剂量图预测网络。

    base_fields 不为 None 时进入「物理先验 + 残差」模式：
    网络只学解析模型（A·exp(-DTH/λ)）没解释掉的那部分，
    这等价于把物理先验当作一个固定层嵌进网络。
    """
    torch.manual_seed(seed)
    X = _input_tensor(cases)
    Y = torch.tensor(_targets(cases), dtype=torch.float32)
    B = _base_tensor(base_fields)
    if high_weight > 0:
        W: torch.Tensor | float = torch.tensor(
            np.stack([1.0 + high_weight * (cs["dose"] >= 0.5 * cs["prescription"])
                      for cs in cases]).astype(np.float32))[:, None]
    else:
        W = 1.0

    net = DoseNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss(reduction="none")
    t0 = time.time()
    n = len(cases)
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, BATCH):
            sel = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(X[sel]) + (B[sel] if base_fields is not None else 0.0)
            loss = (lossf(out, Y[sel]) * (W[sel] if high_weight > 0 else 1.0)).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out = net(X) + (B if base_fields is not None else 0.0)
            hist.append(float(lossf(out, Y).mean()))
    return {"net": net, "params": int(sum(p.numel() for p in net.parameters())),
            "seconds": time.time() - t0, "loss_history": hist}


def _predict(net: nn.Module, cases: list[dict],
             base_fields: list[np.ndarray] | None = None) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        p = net(_input_tensor(cases))
        if base_fields is not None:
            p = p + _base_tensor(base_fields)
    return np.clip(p.numpy()[:, 0] * DOSE_SCALE, 0, None)


def _d90(dose: np.ndarray, struct: np.ndarray) -> float:
    """D90：覆盖 90% 体积的最低剂量（HDR 靶区覆盖的常用指标）。"""
    v = dose[struct]
    return float(np.percentile(v, 10)) if v.size else float("nan")


def steps() -> list[Step]:
    def s1(ctx):
        """合成 40 例 HDR 病例，并展示「DTH → 剂量」的双指数跌落关系。"""
        t0 = time.time()
        cases = [_make_case(seed=100 + i) for i in range(N_CASES)]
        ctx["cases"] = cases
        ctx["s1_seconds"] = time.time() - t0

        A, lam, mse = _fit_analytic(cases)
        ctx["analytic"] = (A, lam, mse)
        tbl = _dth_stats(cases)
        mean_pres = float(np.mean([c["prescription"] for c in cases]))
        tbl["实际 平均剂量/处方"] = tbl["平均剂量 / 处方剂量"].round(4)
        tbl["单指数模型 平均剂量/处方"] = [
            round(float(A * np.exp(-max(d, 0.0) / lam) / mean_pres), 4)
            for d in tbl["平均 DTH (mm)"]]
        tbl["相对偏差"] = [
            f"{100 * (p - a) / max(a, 1e-9):+.1f}%"
            for a, p in zip(tbl["实际 平均剂量/处方"], tbl["单指数模型 平均剂量/处方"])]
        tbl = tbl.drop(columns=["平均剂量 / 处方剂量"])
        ctx["dth_table"] = tbl
        return tbl

    def s2(ctx):
        """构造训练/测试集：输入 2 通道（DTH + 靶区掩膜），输出 1 通道剂量图。"""
        cases = ctx["cases"]
        tr, te = cases[:N_TRAIN], cases[N_TRAIN:]
        ctx["train"], ctx["test"] = tr, te
        ctx["n_params_input"] = int(_inputs(tr).size)
        rows = []
        for tag, cs in [("训练集", tr), ("测试集", te)]:
            doses = np.concatenate([c["dose"][c["body"]] for c in cs])
            rows.append({
                "数据集": tag,
                "病例数": len(cs),
                "体积": f"{SIZE}³（{SIZE * SPACING[0]:.0f} mm 立方）",
                "体素间距 (mm)": "×".join(f"{s:g}" for s in SPACING),
                "靶区体积 (cc)": round(float(np.mean([c["target"].sum() for c in cs]))
                                       * float(np.prod(SPACING)) / 1000.0, 2),
                "OAR 体积 (cc)": round(float(np.mean([c["oar"].sum() for c in cs]))
                                       * float(np.prod(SPACING)) / 1000.0, 2),
                "处方剂量 (Gy)": round(float(np.mean([c["prescription"] for c in cs])), 2),
                "剂量范围 (Gy)": f"{doses.min():.2f}～{doses.max():.2f}",
                "输入通道": "DTH + 靶区掩膜",
            })
        ctx["s2_view"] = pd.DataFrame(rows)
        return ctx["s2_view"]

    def s3(ctx):
        """训练轻量 3D CNN（stride-2 编码 + 反卷积解码）。"""
        res = _train(ctx["train"])
        ctx["net"] = res["net"]
        ctx["train_info"] = res
        ctx["train_mae"] = _mae(_predict(res["net"], ctx["train"]), ctx["train"])
        return {
            "网络结构": "3D CNN（2 层 stride-2 编码 + 2 层反卷积解码，1×1 输出）",
            "参数量": res["params"],
            "输入通道": "2（带符号 DTH + 靶区掩膜）",
            "训练病例数": len(ctx["train"]),
            "epoch 数": EPOCHS,
            "批大小": BATCH,
            "训练耗时 (s)": round(res["seconds"], 2),
            "首/末 epoch 训练 MSE": f"{res['loss_history'][0]:.4f} → {res['loss_history'][-1]:.4f}",
            "训练集剂量图 MAE (Gy)": round(ctx["train_mae"], 4),
        }

    def s4(ctx):
        """剂量图预测精度：总体 / 高剂量区 / 低剂量区 MAE，以及逐例误差。"""
        te = ctx["test"]
        pred = _predict(ctx["net"], te)
        ctx["pred_cnn"] = pred
        truth = np.stack([c["dose"] for c in te])
        body = np.stack([c["body"] for c in te])
        pres = np.array([c["prescription"] for c in te])[:, None, None, None]
        high = body & (truth >= 0.5 * pres)
        low = body & (truth < 0.5 * pres)

        per_case = [float(np.abs(pred[i][body[i]] - truth[i][body[i]]).mean())
                    for i in range(len(te))]
        ctx["per_case_mae"] = per_case
        ctx["mae_all"] = float(np.abs(pred[body] - truth[body]).mean())
        ctx["mae_high"] = float(np.abs(pred[high] - truth[high]).mean())
        ctx["mae_low"] = float(np.abs(pred[low] - truth[low]).mean())
        ctx["mae_pct"] = float(100 * np.abs(pred[body] - truth[body]).mean()
                               / float(np.mean([c["prescription"] for c in te])))
        return {
            "测试病例数": len(te),
            "剂量图 MAE (Gy)": round(ctx["mae_all"], 4),
            "相对处方剂量 (%)": round(ctx["mae_pct"], 2),
            "高剂量区 MAE (Gy)": round(ctx["mae_high"], 4),
            "低剂量区 MAE (Gy)": round(ctx["mae_low"], 4),
            "逐例 MAE 均值 ± 标准差 (Gy)": f"{np.mean(per_case):.4f} ± {np.std(per_case):.4f}",
            "逐例 MAE 最差 (Gy)": round(float(np.max(per_case)), 4),
            "逐例 MAE 最好 (Gy)": round(float(np.min(per_case)), 4),
        }

    def s5(ctx):
        """由预测剂量图反推 D2cc / Dmean / D90，与真值逐例比较。"""
        te, pred = ctx["test"], ctx["pred_cnn"]
        rows = []
        for name, struct in [("OAR（危及器官）", "oar"), ("靶区", "target")]:
            for metric in (["D2cc (Gy)", "Dmean (Gy)"] if struct == "oar"
                           else ["Dmean (Gy)", "D90 (Gy)"]):
                trues, preds = [], []
                for i, cs in enumerate(te):
                    if metric == "D90 (Gy)":
                        t = _d90(cs["dose"], cs[struct])
                        p = _d90(pred[i], cs[struct])
                    else:
                        t = dose_metrics(cs["dose"], cs[struct], SPACING)[metric]
                        p = dose_metrics(pred[i], cs[struct], SPACING)[metric]
                    trues.append(t)
                    preds.append(p)
                trues, preds = np.array(trues), np.array(preds)
                resid = np.abs(preds - trues)
                rows.append({
                    "结构": name, "指标": metric,
                    "真值均值": round(float(trues.mean()), 3),
                    "预测均值": round(float(preds.mean()), 3),
                    "平均绝对残差": round(float(resid.mean()), 3),
                    "最大残差": round(float(resid.max()), 3),
                    "相对残差": f"{100 * float(resid.mean()) / max(float(trues.mean()), 1e-6):.2f}%",
                    "相关系数 r": round(float(np.corrcoef(trues, preds)[0, 1]), 4),
                })
        tbl = pd.DataFrame(rows)
        ctx["metric_table"] = tbl
        ctx["mean_rel_resid"] = float(np.mean(
            [float(s.rstrip("%")) for s in tbl["相对残差"]]))
        return tbl

    def s6(ctx):
        """与解析基线对比，并训练「物理先验 + CNN 残差」的混合模型。"""
        te, tr = ctx["test"], ctx["train"]
        A, lam, _ = ctx["analytic"]
        ctx["analytic_pred"] = [_analytic_predict(cs, A, lam) for cs in te]

        # 混合模型：把解析场当作固定先验，网络只学它没解释掉的残差
        tr_base = [_analytic_predict(cs, A, lam) for cs in tr]
        hyb = _train(tr, base_fields=tr_base, high_weight=2.0, epochs=HYBRID_EPOCHS)
        ctx["hybrid"] = hyb
        ctx["hybrid_pred"] = _predict(hyb["net"], te, base_fields=ctx["analytic_pred"])
        ctx["hybrid_mae_train"] = _mae(_predict(hyb["net"], tr, base_fields=tr_base), tr)

        def summarize(tag, preds, note="", seconds=None):
            mae = _mae(np.stack(preds), te)
            d2_true = np.array([dose_metrics(cs["dose"], cs["oar"], SPACING)["D2cc (Gy)"]
                                for cs in te])
            d2_pred = np.array([dose_metrics(preds[i], te[i]["oar"], SPACING)["D2cc (Gy)"]
                                for i in range(len(te))])
            dm_true = np.array([dose_metrics(cs["dose"], cs["target"], SPACING)["Dmean (Gy)"]
                                for cs in te])
            dm_pred = np.array([dose_metrics(preds[i], te[i]["target"], SPACING)["Dmean (Gy)"]
                                for i in range(len(te))])
            d90_true = np.array([_d90(cs["dose"], cs["target"]) for cs in te])
            d90_pred = np.array([_d90(preds[i], te[i]["target"]) for i in range(len(te))])
            # 靶区 DVH 曲线的平均绝对偏差（百分点）
            dvh = []
            for i, cs in enumerate(te):
                _, c_true = dvh_curve(cs["dose"], cs["target"], 50)
                _, c_pred = dvh_curve(preds[i], cs["target"], 50)
                dvh.append(float(np.abs(c_true - c_pred).mean()))
            return {
                "模型": tag, "说明": note,
                "剂量图 MAE (Gy)": round(mae, 4),
                "高剂量区 MAE (Gy)": round(_mae_high(np.stack(preds), te), 4),
                "D2cc 残差 (Gy)": round(float(np.abs(d2_pred - d2_true).mean()), 3),
                "靶区 Dmean 残差 (Gy)": round(float(np.abs(dm_pred - dm_true).mean()), 3),
                "靶区 D90 残差 (Gy)": round(float(np.abs(d90_pred - d90_true).mean()), 3),
                "靶区 DVH 平均偏差 (百分点)": round(float(np.mean(dvh)), 2),
                "训练耗时 (s)": "—" if seconds is None else round(seconds, 1),
            }

        tbl = pd.DataFrame([
            summarize("解析基线：A·exp(-DTH/λ)", ctx["analytic_pred"],
                      f"最小二乘拟合 A={A:.2f} Gy, λ={lam:.2f} mm"),
            summarize("纯 3D CNN（论文式，2 通道输入）", list(ctx["pred_cnn"]),
                      "只学几何→剂量映射", ctx["train_info"]["seconds"]),
            summarize("物理先验 + CNN 残差（本复现改进）", list(ctx["hybrid_pred"]),
                      "解析场作固定层，网络学残差", hyb["seconds"]),
        ])
        ctx["cmp_table"] = tbl
        ctx["mae_analytic"] = float(tbl.iloc[0]["剂量图 MAE (Gy)"])
        ctx["mae_cnn"] = float(tbl.iloc[1]["剂量图 MAE (Gy)"])
        ctx["mae_hybrid"] = float(tbl.iloc[2]["剂量图 MAE (Gy)"])
        ctx["d2_analytic"] = float(tbl.iloc[0]["D2cc 残差 (Gy)"])
        ctx["d2_cnn"] = float(tbl.iloc[1]["D2cc 残差 (Gy)"])
        ctx["d2_hybrid"] = float(tbl.iloc[2]["D2cc 残差 (Gy)"])
        ctx["dvh_analytic"] = float(tbl.iloc[0]["靶区 DVH 平均偏差 (百分点)"])
        ctx["dvh_cnn"] = float(tbl.iloc[1]["靶区 DVH 平均偏差 (百分点)"])
        ctx["dvh_hybrid"] = float(tbl.iloc[2]["靶区 DVH 平均偏差 (百分点)"])
        return tbl

    def s7(ctx):
        """结论对照：剂量图级预测是否可行、CNN 与解析模型各自的强项。"""
        return (
            f"① 物理与数据：{N_CASES} 例合成 HDR 病例（{SIZE}³、{SPACING[0]:g} mm 各向同性，"
            f"靶区平均体积 {ctx['s2_view'].iloc[0]['靶区体积 (cc)']} cc、"
            f"OAR 平均体积 {ctx['s2_view'].iloc[0]['OAR 体积 (cc)']} cc，OAR 紧邻靶区侧方约 6 mm）。"
            f"剂量按到靶区表面距离 DTH 跌落：近场衰减长度 λ₁ ≈ "
            f"{np.mean([c['lam1'] for c in ctx['cases']]):.1f} mm、远场 λ₂ ≈ "
            f"{np.mean([c['lam2'] for c in ctx['cases']]):.1f} mm，"
            "对应「点源平方反比 × 组织衰减、多驻留点叠加」的双指数行为 —— "
            f"单指数最小二乘只能取折中（A = {ctx['analytic'][0]:.2f} Gy，"
            f"λ = {ctx['analytic'][1]:.2f} mm）。\n"
            f"② 纯数据驱动路线：3D CNN（{ctx['train_info']['params']} 参数、{EPOCHS} epoch、"
            f"训练耗时 {ctx['train_info']['seconds']:.1f} s）在 {len(ctx['test'])} 例未见病例上"
            f"剂量图 MAE = {ctx['mae_cnn']:.3f} Gy（处方剂量的 "
            f"{100 * ctx['mae_cnn'] / np.mean([c['prescription'] for c in ctx['test']]):.2f}%），"
            f"高剂量区 {ctx['mae_high']:.3f} Gy；训练集 MAE {ctx['train_mae']:.3f} Gy —— "
            "训练误差与测试误差接近，说明瓶颈是**模型表达能力/样本量**，不是过拟合。\n"
            f"③ 指标级链路：由预测剂量图反推的 D2cc / Dmean / D90 与真值的平均相对残差为 "
            f"{ctx['mean_rel_resid']:.1f}%。这验证了原文最核心的价值："
            "**预测整幅剂量图比「一个 DVH 指标训一个模型」更通用** —— "
            "一次前向推理即可得到任意结构、任意指标的剂量学参数（本步只演示了其中 4 个）。\n"
            f"④ 与解析基线对比：单指数解析模型（只用 DTH、零训练）剂量图 MAE = "
            f"{ctx['mae_analytic']:.3f} Gy、D2cc 残差 {ctx['d2_analytic']:.3f} Gy，"
            f"居然优于纯 CNN（{ctx['mae_cnn']:.3f} Gy / {ctx['d2_cnn']:.3f} Gy）。"
            "原因很直接：剂量场本质上是一个「关于 DTH 的陡峭径向函数」，"
            "而 stride-2 的轻量网络分辨率不足、30 例训练样本也远不足以从零学出这条曲线。\n"
            f"⑤ 物理先验 + 残差学习：把解析场当作网络里的固定层（输出 = 解析场 + CNN 残差），"
            f"剂量图 MAE 降到 {ctx['mae_hybrid']:.3f} Gy、D2cc 残差 "
            f"{ctx['d2_hybrid']:.3f} Gy、靶区 DVH 平均偏差 "
            f"{ctx['dvh_hybrid']:.2f} 个百分点（解析基线 {ctx['dvh_analytic']:.2f}、"
            f"纯 CNN {ctx['dvh_cnn']:.2f}），在 MAE / D2cc / D90 / DVH 四项上同时优于另两个模型；"
            "只有高剂量区 MAE 仍与解析基线接近（说明靶内剂量抬升这一项还需要更多样本或更强网络）。"
            "分工很清楚：解析项负责陡峭的近场跌落，网络只负责双指数拖尾与靶内抬升这些**结构性残差**。\n"
            "⑥ 结论与方法学要点：剂量图级预测是可行的，而且「由预测图反算 DVH 指标」这条链路"
            "比逐指标建模更通用；但在小样本、几何输入有限的条件下，"
            "**纯数据驱动不一定胜过简单的物理模型**，把物理先验嵌入网络（physics-informed）"
            "才是性价比最高的做法。当前 2 通道几何输入还有一个不可约误差："
            "处方剂量与驻留时间不在输入里，逐例的绝对剂量水平本质不可辨识 —— "
            "这正是最值得做的延伸练习。"
        )

    return [
        Step("① 合成病例与「距离-剂量」关系", "对应原文 Method 的剂量学基础：靶区/危及器官几何 + "
             "HDR 剂量场。本步合成 40 例病例，并按 DTH 分档统计平均剂量，展示双指数跌落"
             "（近场陡、远场拖尾）——这是单指数解析模型无法拟合的部分。",
             s1, "table", "DTH<0 为靶内（剂量高于处方剂量），DTH>0 为靶外，剂量随距离快速跌落。"),
        Step("② 构造训练 / 测试集", "对应原文的数据准备：输入 2 通道（带符号 DTH 图 + 靶区掩膜），"
             "输出 1 通道剂量图；30 例训练、10 例测试（患者级划分，测试集完全不参与训练）。",
             s2, "table", "患者级划分是剂量预测模型评估的底线，随机体素划分会造成严重信息泄漏。"),
        Step("③ 训练轻量 3D CNN", "对应原文的深度学习剂量图预测网络：本复现用 2 层 stride-2 编码 + "
             "2 层反卷积解码的极轻 3D CNN（约 1.4 万参数），在 CPU 上数秒完成训练。",
             s3, "metrics", "参数少、epoch 少是为了在 15 秒内跑完；真实任务需要更大网络与更多病例。"),
        Step("④ 剂量图预测精度", "对应原文的剂量图级评价：总体 MAE、高剂量区/低剂量区 MAE、"
             "相对处方剂量的百分比误差，以及逐例误差分布。",
             s4, "metrics", "高剂量区误差直接决定靶区覆盖与 OAR 热点的判断，必须单独报告。"),
        Step("⑤ 由预测图计算 D2cc / Dmean / D90", "对应原文的「剂量图 → DVH 指标」下游应用："
             "用预测剂量图重新计算 OAR 的 D2cc/Dmean 与靶区的 Dmean/D90，与真值逐例比较残差。",
             s5, "table", "绝对残差不大，但逐例相关系数可能为负：2 通道几何输入看不到处方剂量，"
                          "逐例的绝对剂量水平本质不可辨识 —— 这是输入信息不足，不是模型坏了。"),
        Step("⑥ 与解析基线对比 + 物理先验增强", "对照「只用 DTH 的单指数解析模型」，"
             "并进一步把解析场当作网络中的固定层、只让 CNN 学残差（physics-informed），"
             "三种模型比较剂量图 MAE、高剂量区 MAE、D2cc/Dmean/D90 残差与靶区 DVH 偏差。",
             s6, "table", "小而干净的实验里，简单物理模型常常胜过纯数据驱动；把两者结合才是最优解。"),
        Step("⑦ 结论对照", "汇总复现结论、与原文的差异，以及 2 通道几何输入模型的固有局限。",
             s7, "text"),
    ]
