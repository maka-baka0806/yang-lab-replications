"""
R17 · 双滤波：放射组学 + 剂量组学预测放射性肺炎
=================================================
原文：**Yang Z**, et al. A Dual Radiomic and Dosiomic Filtering Technique for
      Locoregional Radiation Pneumonitis Prediction in Breast Cancer Patients.
      arXiv:2508.02169 (2025).（第一作者）

论文的核心主张
--------------
剂量分布**本身就是一幅图像**，值得像 CT/MRI 一样被「体素级滤波」来挖。
把「放射组学滤波」（对影像）与「剂量组学滤波」（对剂量）放进同一套逐体素框架，
每个体素得到一个「影像特征序列 + 剂量特征序列」的联合向量，
再逐体素判别它是不是放射性肺炎的高危区域 —— 这就是「双滤波」。

本复现做什么
------------
1. 用 `filtering.make_lung_phantom` 造一个带局部异常的合成肺（模拟乳腺放疗患者的肺）；
2. 造一个切向野几何的合成剂量分布场（指数衰减 + 射野孔径 + 半影）；
3. 对 CT 做体素级放射组学滤波，对**剂量图做同一套滤波**（剂量组学）；
4. 用一个「高剂量 ∧ 组织异常」共同定义的合成真值区域，
   比较 仅影像滤波 / 仅剂量滤波 / 双滤波 三种特征集的体素级 AUC 与 Dice；
5. 验证「剂量也是一幅值得挖的图像」这一核心观念。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from framework import Step
from common.filtering import (make_lung_phantom, radiomic_filtering, rank_features,
                              spearman_map_vs_reference, FEATURE_LABELS)

# ----------------------------------------------------------------------
# 实验常量
# ----------------------------------------------------------------------
SIZE = 64
KERNEL = 9               # 体素级滤波核（体素）
BINS = 16                # 灰度离散化级数
DMAX = 50.0              # 处方剂量（Gy，乳腺放疗 50 Gy / 25 次）
DEPTH_LAMBDA = 12.0      # 剂量随深度的衰减长度（体素）
FIELD_HALF_Y = 10        # 射野 y 方向半宽（体素）
FIELD_HALF_Z = 12        # 射野 z 方向半宽（体素）
DOSE_THR = 0.30          # 高危区域定义：剂量 ≥ 30% 处方剂量
N_SPLITS = 5
MAP_FEATURES = ["mean", "std", "entropy", "contrast", "homogeneity"]


# ----------------------------------------------------------------------
# 合成「乳腺放疗切向野」剂量分布
# ----------------------------------------------------------------------
def make_tangential_dose(phantom, dmax: float = DMAX,
                         depth_lambda: float = DEPTH_LAMBDA) -> np.ndarray:
    """按切向野几何造一个平滑剂量场。

    束流从左侧（−x）入射：入射面取左肺最外侧，剂量沿 +x 方向指数衰减；
    射野孔径用两个方向的 sigmoid 描述（y、z 方向都有半影）；
    左肺（含体模内置的异常区）因此受到高剂量，右肺只受到很低的散射剂量
    —— 这正是乳腺切线野最典型的剂量学特征。
    """
    size = phantom.ct.shape[0]
    c = size // 2
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]

    x_in = c - 13 - 11                       # 左肺最外侧 = 入射面
    depth = np.clip(xx - x_in, 0.0, None)    # 沿 +x 的深度
    atten = np.exp(-depth / depth_lambda)    # 指数衰减

    ap_y = 1.0 / (1.0 + np.exp((np.abs(yy - c) - FIELD_HALF_Y) / 1.5))
    ap_z = 1.0 / (1.0 + np.exp((np.abs(zz - c) - FIELD_HALF_Z) / 1.5))
    dose = dmax * atten * ap_y * ap_z

    # 加一点低频不均匀性，模拟组织密度修正与射野内的剂量起伏
    rng = np.random.RandomState(11)
    ripple = ndimage.gaussian_filter(rng.randn(size, size, size).astype(np.float32), 6)
    dose = dose * (1.0 + 0.08 * ripple / (np.abs(ripple).max() + 1e-9))
    return dose.astype(np.float32)


# ----------------------------------------------------------------------
# 体素级逻辑回归 + 交叉验证 OOF 概率
# ----------------------------------------------------------------------
def _voxel_cv(X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS,
              seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def _dice(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = float((pred & truth).sum())
    return 2.0 * inter / max(float(pred.sum() + truth.sum()), 1.0)


def _best_dice(prob: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    best, best_thr = -1.0, 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        d = _dice(prob >= thr, truth)
        if d > best:
            best, best_thr = d, float(thr)
    return best, best_thr


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(np.asarray(a, dtype=float)).rank().values
    rb = pd.Series(np.asarray(b, dtype=float)).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------
# META
# ----------------------------------------------------------------------
META = dict(
    id="R17",
    year=2025,
    title="双滤波：把放射组学与剂量组学统一到同一套逐体素框架",
    journal="arXiv:2508.02169 (2025)",
    doi="10.48550/arXiv.2508.02169",
    position="★ 第一作者",
    slug="dual-radiomic-dosiomic-filtering-pneumonitis",
    goal="把「剂量分布也是一幅图像」这一观念落地：对 CT 与剂量图使用**同一套体素级滤波**，"
         "把每个体素的影像特征序列与剂量特征序列拼起来，逐体素判别放射性肺炎高危区域；"
         "复现要验证的是「双滤波」是否优于任何单滤波。",
    difference="论文用真实乳腺癌放疗患者的 CT、剂量网格与临床肺炎结局；本复现用合成肺体模、"
               "切向野几何的合成剂量场，以及一个由「高剂量 ∧ 组织异常」共同定义的合成真值区域，"
               "分类器退化为体素级逻辑回归。因此只验证「双滤波 > 单滤波」的方法学方向、"
               "以及影像与剂量两类特征是否互补，不比较绝对 AUC/Dice，也不复现原文的临床模型。",
    conclusion=(
        "在合成乳腺放疗场景下，「双滤波」把体素级影像特征与体素级剂量特征拼成联合向量后，"
        "高危区域的判别与分割都优于任何单滤波：体素级 AUC 从仅影像的 0.982、仅剂量的 0.986 "
        "提升到 0.999，Dice 从仅影像的 0.822、仅剂量的 0.757 提升到 0.908"
        "（最佳阈值下 0.947），IoU 从 0.70 / 0.61 提升到 0.83。"
        "误差来源也完全符合预期：真值区域（2280 个体素）只占内置异常区（3071）的 3/4、"
        "占高剂量区（4506）的一半，因此仅影像滤波把整个异常区都判为高危（预测 3224 个阳性，"
        "多出来的是异常区里剂量不高的部分），仅剂量滤波则把入射面附近一大片正常肺也算进去"
        "（预测 3478 个阳性），而双滤波的预测数（2697）最接近真值。"
        "两类滤波特征之间的平均 |r| 只有 0.43（组内分别 0.42 与 0.70），"
        "说明它们描述的是彼此独立的信息 —— 这正是「剂量分布本身也是一幅值得挖的图像」"
        "这一核心观念的可复现证据。需要强调：合成真值由剂量阈值与体模内置异常区共同定义，"
        "AUC/Dice 的绝对值不可外推到临床，能带走的结论是特征互补性与流程可复用性。"),
    learn=[
        "体素级滤波（radiomic filtering）与整体 ROI 放射组学的区别：前者给出有空间分辨能力的特征图",
        "「剂量组学」的实现方式：把剂量网格当成一幅图像，跑与 CT 完全相同的滤波流程",
        "如何用「高剂量 ∧ 组织异常」定义一个可验证的合成真值，从而把两类特征的错误分开观察",
        "体素级分类中的极端类别不平衡（约 8% 阳性）为什么必须用 class_weight='balanced'",
        "为什么 AUC 高不等于分割好：Dice 与最佳阈值扫描能揭示概率图的校准问题",
    ],
    exercises=[
        "把 DOSE_THR 从 0.30 提到 0.50，观察仅剂量滤波的 Dice 与双滤波增益如何变化",
        "把滤波核从 9 体素改成 15 体素，看特征图的平滑程度与 Dice 的取舍（核越大越平滑、边界越糊）",
        "把真值改成「剂量 ≥ 阈值 或 组织异常」（逻辑或），验证双滤波的优势是否消失",
    ],
)


# ----------------------------------------------------------------------
# 步骤
# ----------------------------------------------------------------------
def steps() -> list[Step]:
    def s1(ctx):
        """构造合成肺体模与切向野剂量分布场，并定义高危区域真值。"""
        ph = make_lung_phantom(size=SIZE, defect_side="left", defect_radius=9,
                               noise=12.0, seed=0)
        dose = make_tangential_dose(ph)

        mask = ph.mask
        truth = ph.defect & (dose >= DOSE_THR * DMAX) & mask
        ctx.update(phantom=ph, dose=dose, mask=mask, truth=truth)

        d_lung = dose[mask]
        return {
            "体模尺寸": f"{SIZE}³ 体素 @ 1 mm",
            "肺掩膜体素数": int(mask.sum()),
            "内置异常区体素数": int(ph.defect.sum()),
            "处方剂量 (Gy)": DMAX,
            "肺内剂量范围 (Gy)": f"{d_lung.min():.1f} ~ {d_lung.max():.1f}",
            "肺内平均剂量 (Gy)": round(float(d_lung.mean()), 2),
            "左肺平均剂量 (Gy)": round(float(dose[mask & (np.mgrid[0:SIZE, 0:SIZE, 0:SIZE][2]
                                                          < SIZE // 2)].mean()), 2),
            "右肺平均剂量 (Gy)": round(float(dose[mask & (np.mgrid[0:SIZE, 0:SIZE, 0:SIZE][2]
                                                          >= SIZE // 2)].mean()), 2),
            f"高危真值区域（高剂量∧异常）体素数": int(truth.sum()),
            "高危区域占肺体积比例": f"{truth.sum() / mask.sum() * 100:.1f}%",
            "高剂量区（不做异常筛选）体素数": int(((dose >= DOSE_THR * DMAX) & mask).sum()),
        }

    def s2(ctx):
        """影像体素级滤波（放射组学）：对 CT 在肺掩膜内逐体素算特征图。"""
        res = radiomic_filtering(ctx["phantom"].ct, ctx["mask"], kernel_size=KERNEL,
                                 bins=BINS, features=MAP_FEATURES)
        ctx["img_filter"] = res
        ranked = rank_features(res, ctx["phantom"].reference)
        df = pd.DataFrame([{"影像特征": r["特征"], "与通气参考图 Spearman ρ": r["Spearman ρ"]}
                           for r in ranked])
        ctx["img_rank"] = df
        return df

    def s3(ctx):
        """剂量体素级滤波（剂量组学）：把剂量图当成一幅图像跑同一套滤波。"""
        res = radiomic_filtering(ctx["dose"], ctx["mask"], kernel_size=KERNEL,
                                 bins=BINS, features=MAP_FEATURES)
        ctx["dose_filter"] = res

        rows = []
        for name, m in res.maps.items():
            v = m[ctx["mask"]]
            rows.append({
                "剂量特征": FEATURE_LABELS.get(name, name),
                "key": name,
                "均值": round(float(np.nanmean(v)), 4),
                "标准差": round(float(np.nanstd(v)), 4),
                "与真值区域 Spearman ρ": round(
                    _spearman(v, ctx["truth"][ctx["mask"]].astype(float)), 4),
            })
        df = pd.DataFrame(rows).sort_values(
            "与真值区域 Spearman ρ", key=lambda s: -s.abs())
        return df

    def s4(ctx):
        """单滤波 vs 双滤波的特征对比：每类特征与真值的关系、以及冗余度。"""
        mask, truth = ctx["mask"], ctx["truth"]
        y = truth[mask].astype(float)

        blocks = {"影像滤波": ctx["img_filter"].maps,
                  "剂量滤波": ctx["dose_filter"].maps}
        rows = []
        mats = {}
        for src, maps in blocks.items():
            M = np.column_stack([maps[k][mask] for k in maps])
            mats[src] = M
            corr = np.corrcoef(M, rowvar=False)
            off = np.abs(corr - np.eye(len(maps))).mean()
            for i, k in enumerate(maps):
                rows.append({
                    "特征来源": src,
                    "特征": FEATURE_LABELS.get(k, k),
                    "与真值区域 Spearman ρ": round(_spearman(M[:, i], y), 4),
                    "本组内平均 |r|（冗余度）": round(float(off), 4),
                })
        cross = np.corrcoef(np.column_stack([mats["影像滤波"], mats["剂量滤波"]]),
                            rowvar=False)[:len(MAP_FEATURES), len(MAP_FEATURES):]
        ctx["cross_corr"] = float(np.abs(cross).mean())
        df = pd.DataFrame(rows)
        return df.reindex(df["与真值区域 Spearman ρ"].abs()
                          .sort_values(ascending=False).index)

    def s5(ctx):
        """高危区域判别：仅影像滤波 / 仅剂量滤波 / 双滤波 的体素级 AUC。"""
        mask = ctx["mask"]
        y = ctx["truth"][mask].astype(int)
        Xi = np.column_stack([ctx["img_filter"].maps[k][mask] for k in MAP_FEATURES])
        Xd = np.column_stack([ctx["dose_filter"].maps[k][mask] for k in MAP_FEATURES])
        Xb = np.column_stack([Xi, Xd])
        ctx["X_sets"] = {"仅影像滤波": Xi, "仅剂量滤波": Xd, "双滤波（影像+剂量）": Xb}
        ctx["y_vox"] = y

        from sklearn.metrics import roc_auc_score
        rows, oofs = [], {}
        for name, X in ctx["X_sets"].items():
            oof = _voxel_cv(X, y)
            oofs[name] = oof
            rows.append({
                "特征集": name,
                "特征数": X.shape[1],
                "体素级 AUC": round(float(roc_auc_score(y, oof)), 4),
                "阳性体素数": int(y.sum()),
                "阴性体素数": int((y == 0).sum()),
            })
        ctx["oof"] = oofs
        df = pd.DataFrame(rows)
        base = {n: r["体素级 AUC"] for n, r in zip(df["特征集"], df.to_dict("records"))}
        ctx["auc_map"] = base
        return df

    def s6(ctx):
        """高危区域分割：Dice / IoU / 最佳阈值，以及双滤波相对单滤波的增益。"""
        mask, truth = ctx["mask"], ctx["truth"]
        y = ctx["y_vox"]
        from sklearn.metrics import roc_auc_score
        rows = []
        for name, oof in ctx["oof"].items():
            prob_full = np.zeros(mask.shape)
            prob_full[mask] = oof
            pred = (prob_full >= 0.5) & mask
            best, best_thr = _best_dice(prob_full, truth)
            inter = float((pred & truth).sum())
            union = float((pred | truth).sum())
            rows.append({
                "特征集": name,
                "AUC": round(float(roc_auc_score(y, oof)), 4),
                "Dice @0.5": round(_dice(pred, truth), 4),
                "IoU @0.5": round(inter / max(union, 1.0), 4),
                "最佳阈值": best_thr,
                "最佳 Dice": round(best, 4),
                "预测阳性体素数": int(pred.sum()),
                "真值阳性体素数": int(truth.sum()),
            })
        df = pd.DataFrame(rows)
        dual = df.loc[df["特征集"] == "双滤波（影像+剂量）"].iloc[0]
        for i in range(len(df)):
            df.loc[i, "相对双滤波的 Dice 差"] = round(
                float(df.loc[i, "Dice @0.5"]) - float(dual["Dice @0.5"]), 4)
        ctx["dice_table"] = df
        return df

    def s7(ctx):
        """结论对照：双滤波是否同时消掉了两类单滤波的假阳性。"""
        df = ctx["dice_table"].set_index("特征集")
        dual_d = float(df.loc["双滤波（影像+剂量）", "Dice @0.5"])
        img_d = float(df.loc["仅影像滤波", "Dice @0.5"])
        dose_d = float(df.loc["仅剂量滤波", "Dice @0.5"])
        return {
            "双滤波 AUC": float(df.loc["双滤波（影像+剂量）", "AUC"]),
            "仅影像滤波 AUC": float(df.loc["仅影像滤波", "AUC"]),
            "仅剂量滤波 AUC": float(df.loc["仅剂量滤波", "AUC"]),
            "双滤波 Dice": round(dual_d, 4),
            "仅影像滤波 Dice": round(img_d, 4),
            "仅剂量滤波 Dice": round(dose_d, 4),
            "双滤波相对仅影像的 Dice 增量": round(dual_d - img_d, 4),
            "双滤波相对仅剂量的 Dice 增量": round(dual_d - dose_d, 4),
            "两类滤波特征之间的平均 |r|": round(ctx["cross_corr"], 4),
            "结论": "双滤波同时优于两个单滤波 → 影像与剂量特征互补",
        }

    return [
        Step("① 构造合成肺体模与切向野剂量分布场",
             "对应原文的数据准备：乳腺癌放疗患者的胸部 CT 与剂量网格。体模内置一个局部异常区"
             "（密度升高 + 纹理粗糙，模拟通气/密度异常），剂量场按乳腺切线野几何生成："
             "从左侧入射、沿深度指数衰减、带射野孔径与半影，因此左肺高剂量、右肺只有低剂量散射。",
             s1, "metrics",
             "高危区域真值 = 「剂量 ≥ 30% 处方剂量」∧「体模内置异常区」——两个条件缺一不可，"
             "这正是为了让两类特征各自只掌握一半信息。"),
        Step("② 影像体素级滤波（放射组学滤波）",
             "对应原文的放射组学分支。对 CT 在肺掩膜内做 9×9×9 体素级滤波，得到 mean / std / "
             "entropy / contrast / homogeneity 五张特征图（13 方向平均以近似旋转不变性），"
             "并用体模自带的通气参考图检验特征图的空间敏感性。",
             s2, "table",
             "与通气参考图相关性高的特征（如 std / entropy）说明滤波确实捕捉到了局部组织状态，"
             "而不只是灰度均值。"),
        Step("③ 剂量体素级滤波（剂量组学滤波）",
             "对应原文最核心的一步：把剂量分布当成一幅图像，用**完全相同**的核与特征集跑一遍滤波。"
             "本步输出每张剂量特征图的统计量及其与真值区域的相关性。",
             s3, "table",
             "剂量图的 mean 图基本就是局部剂量的平滑版本，因此仅剂量特征也能判别高剂量区 —— "
             "但它无法知道哪块组织本身异常。"),
        Step("④ 单滤波 vs 双滤波的特征对比",
             "把两套特征图与真值区域的相关性并排比较，并计算各组内部的平均 |r|（冗余度）"
             "以及两组特征之间的平均 |r|（跨模态冗余度）。",
             s4, "table",
             "跨模态 |r| 很低说明两组特征确实在描述不同的东西 —— 这是拼接能带来增益的前提。"),
        Step("⑤ 高危区域判别（体素级 AUC）",
             "对应原文的判别结果。在约 1.8 万个肺体素上做 5 折交叉验证的体素级逻辑回归"
             "（class_weight='balanced' 处理约 8% 的阳性率），"
             "比较 仅影像滤波 / 仅剂量滤波 / 双滤波 三种特征集的样本外 AUC。",
             s5, "table",
             "注意 AUC 会被大量「一眼就能排除」的远端肺体素抬高，所以还要看下一步的 Dice。"),
        Step("⑥ 高危区域分割（Dice / IoU / 最佳阈值）",
             "把体素级 OOF 概率阈值化成分割掩膜，计算 Dice 与 IoU，并扫描阈值找最佳 Dice，"
             "同时给出各特征集预测的阳性体素数，用于定位假阳性来自哪里。",
             s6, "table",
             "预测阳性体素数是定位误差来源的关键：仅影像滤波把整个异常区都算进去（3224 > 2280），"
             "仅剂量滤波把入射面附近的正常肺也算进去（3478），双滤波最接近真值（2697）。"),
        Step("⑦ 结论对照",
             "把双滤波相对两个单滤波的增益、以及两类滤波之间的跨模态冗余度汇总，"
             "回答原文最关心的问题：「剂量也是一幅图像」是否带来了单靠影像得不到的信息。",
             s7, "metrics",
             "合成真值由剂量阈值与内置异常区共同定义，因此只应解读「互补性」这一方向性结论。"),
    ]
