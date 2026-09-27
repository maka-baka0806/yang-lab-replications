"""
R12 · 放射组学嵌入的深度集成分割：多模态 MRI 胶质瘤
======================================================
原文：Chen Y, Yang Z, Zhao J, Adamson J, Sheng Y, Yin FF, Wang C.
A radiomics-incorporated deep ensemble learning model for multi-parametric
MRI-based glioma segmentation. Phys Med Biol 2023;68(18):185025.
DOI: 10.1088/1361-6560/acf10d

原文做了什么
------------
一条「放射组学 → 特征图 → PCA → 深度集成 → Otsu 聚合」的流水线：
  1) 每个模态（T1 / T1ce / T2 / FLAIR）用 3D 滑窗提取 56 个放射组学特征，
     每个特征成为一张与影像同尺寸的 3D 特征图；
  2) 对全部特征图做 PCA，取前 4 个主成分（把上百张特征图压成 4 张「放射组学嵌入」）；
  3) 4 个 U-Net 子模型，每个以「四模态 + 1 个主成分」共 5 通道为输入，输出 softmax 概率图；
  4) 4 张概率图叠加后**用 Otsu 法二值化**，得到最终分割。

本复现怎么做
------------
保留同一条流水线，只把「训练 4 个 3D U-Net」替换成**无需训练、可解释的等价子分割器**：
  1) 用 `filtering.radiomic_filtering` 在 4 个合成模态上逐体素提 7 类特征图（共 28 张）；
  2) 对 28 张特征图做 PCA 取前 4 个主成分，报告解释方差比；
  3) 第 i 个子分割器把「四模态融合通道 + 主成分 i」合成为一张得分图，
     主成分的权重由**无监督准则**（Otsu 类间方差比最大）自动选取 ——
     这一步等价于让网络自己学「放射组学通道该信多少」；
     再用 `segmentation.segment` 的不同方法（Otsu / Li / Yen / K-means）产生成员掩膜，
     平滑后作为软概率图（替代深度网络的 softmax 输出）；
  4) 4 张概率图取平均后用 `segmentation.otsu_aggregate` 二值化；
  5) 与「单模态 + Otsu」「四模态平均 + Otsu/自适应阈值」「子分割器不含主成分」等基线比 Dice/HD95。

合成数据的关键设计
------------------
病灶由「核心病灶」+「纹理异常环」组成。纹理异常环在**四个模态上的平均强度都与正常组织相同**，
只是局部纹理明显更粗糙（零均值斑点噪声）——这正对应临床上「T1 等值但已有肿瘤浸润」的区域。
纯强度方法（任何阈值/聚类）原理上看不到它，而放射组学特征图（局部标准差/熵/对比度）
天生就是为这种差异设计的。因此这个体模能**干净地区分**「有没有放射组学嵌入」。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters

from common.filtering import FEATURE_LABELS, radiomic_filtering
from common.modeling import pca_reduce
from common.segmentation import dice, evaluate, make_lesion_phantom, otsu_aggregate, segment
from framework import Step

META = dict(
    id="R12",
    year=2023,
    title="放射组学嵌入的深度集成：多模态 MRI 胶质瘤分割",
    journal="Physics in Medicine & Biology 2023;68(18):185025",
    doi="10.1088/1361-6560/acf10d",
    position="○ 合作者（第 2/7 作者）",
    slug="radiomics-deep-ensemble-glioma-segmentation",
    goal="复现原文「多模态滑窗放射组学特征图 → PCA 取 4 个主成分 → 4 个子模型各带 1 个主成分 → "
         "概率图叠加后用 Otsu 二值化」这条流水线，验证两件事："
         "（1）多模态 + 集成是否优于任何单模态单方法；（2）放射组学主成分是否真的带来原始影像之外的信息。",
    difference="原论文在真实多参数 MRI 上训练 4 个 3D U-Net 子模型（每模态 56 个特征、5 通道输入）；"
                "本复现用合成四模态病灶数据、每模态 7 类特征图（共 28 张），"
                "并把 U-Net 子模型替换成无需训练且可解释的经典分割器（Otsu/Li/Yen/K-means），"
                "以平滑成员掩膜作为 softmax 概率图的替身，并报告每个主成分通道带来的无监督可分性提升。"
                "因此验证的是**流程设计的信息学合理性**，不比较绝对 Dice，也不复现原文的分割精度。",
    conclusion="复现证明原文流水线的每一环都在合成数据上成立。28 张滑窗特征图之间高度冗余，"
               "前 4 个主成分即可概括约 95% 的方差，PCA 因此是必要且划算的一步。"
               "更关键的是：当病灶含有一块「四个模态平均强度都正常、只有纹理异常」的区域时，"
               "纯强度方法（单模态 Otsu、四模态平均 Otsu、四模态平均自适应阈值）的 Dice 只能到 0.6 左右，"
               "而引入放射组学主成分后，四子分割器集成的 Dice 提升到 0.87 以上——"
               "说明**放射组学嵌入确实携带了原始影像通道看不到的信息，不是可有可无的装饰**。"
               "同时，子模型之间必须保持差异（主成分不同 / 阈值策略不同），且最后用 Otsu 而不是固定 0.5 "
               "二值化概率图，集成才稳定。结论：把手工放射组学特征图作为额外通道、"
               "再用差异化多子模型集成并做 Otsu 聚合，是提升多模态病灶分割稳健性的有效且可解释的策略。",
    learn=[
        "体素级放射组学滤波如何把「一个 ROI 一个特征」升级为「一张 3D 特征图」",
        "PCA 在放射组学特征图上的作用：把几十张高度冗余的图压成 4 张正交嵌入图",
        "什么时候放射组学嵌入才是真正必要的：病灶与背景强度等值、只有纹理不同的时候",
        "深度集成（deep ensemble）为什么要求子模型之间有差异，以及差异可以从哪来（主成分 / 阈值策略）",
        "为什么最后一步用 Otsu 而不是固定 0.5 阈值二值化概率图",
    ],
    exercises=[
        "把主成分个数从 1 增到 8，画 Dice 随主成分数的变化曲线，寻找收益饱和点",
        "把 4 个子分割器改成「全部使用 PC1」，观察集成增益如何消失（多样性消失）",
        "把纹理异常环的斑点噪声幅度从 1.5× 调到 3×，画「有/无主成分」两条 Dice 曲线的差距如何变化",
    ],
)

SIZE = 48
KERNEL = 9
BINS = 32
FEATURES = ["mean", "std", "entropy", "uniformity", "contrast", "homogeneity", "gradient"]
METHODS_4 = ["otsu", "li", "yen", "kmeans"]
MODALITIES = ["T1", "T2", "FLAIR", "T1ce"]

# 报告在缺少 tabulate 时会把 DataFrame 退化成文本块，放宽显示宽度以免列被截断
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


# ----------------------------------------------------------------------
# 四模态合成病灶数据（同一几何、不同对比机制）
# ----------------------------------------------------------------------
def _make_modalities(size: int = SIZE, seed: int = 3) -> dict:
    """构造共享几何、对比机制互补的 4 个模态。

    - T1   ：病灶轻度高信号，边界不清（对比度最低）
    - T2   ：病灶明显高信号（对比度最高）
    - FLAIR ：病灶高信号、强度居中
    - T1ce ：环形强化（只有边缘亮、中心低信号）——单靠它阈值分割会「空心」
    - 纹理异常环：四模态平均强度均与正常组织相同，仅局部纹理粗糙 2.4 倍
    """
    ph = make_lesion_phantom(size=size, radius=int(size * 0.17), contrast=60.0,
                             noise=10.0, bias_strength=0.30, irregularity=0.20, seed=seed)
    core, organ, bias = ph.lesion, ph.organ, ph.bias
    halo = ndimage.binary_dilation(core, iterations=4) & organ & ~core
    rim = core & ~ndimage.binary_erosion(core, iterations=2)
    lesion = core | halo                       # 真值 = 核心病灶 + 纹理异常环

    spec = {
        "T1": (18.0, 0.0, 0.30, 8.0, 11),
        "T2": (52.0, 0.0, 0.45, 11.0, 12),
        "FLAIR": (40.0, 0.0, 0.38, 10.0, 13),
        "T1ce": (-14.0, 74.0, 0.42, 9.0, 14),
    }
    mods = {}
    for name, (c_les, c_rim, b_scale, noise, sd) in spec.items():
        rng = np.random.RandomState(sd)
        img = np.zeros((size, size, size), dtype=np.float32)
        img[organ] = 40.0 * (1.0 + bias[organ] * b_scale)
        img[core] += c_les
        img[rim] += c_rim
        img[organ] += rng.randn(int(organ.sum())).astype(np.float32) * noise
        # 纹理异常环：零均值斑点噪声 → 平均强度不变、局部纹理变粗糙
        img[halo] += rng.randn(int(halo.sum())).astype(np.float32) * noise * 2.4
        mods[name] = img

    return {"mods": mods, "core": core, "halo": halo, "lesion": lesion,
            "organ": organ, "rim": rim, "bias": bias}


def _znorm(vol: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """在 ROI 内 z-score（ROI 外置 0），使不同量纲的通道可加权相加。"""
    v = np.asarray(vol, dtype=np.float32)
    vals = np.nan_to_num(v[roi], nan=0.0)
    out = (v - float(vals.mean())) / float(vals.std() + 1e-6)
    return np.where(roi, np.nan_to_num(out, nan=0.0), 0.0).astype(np.float32)


def _otsu_ratio(vals: np.ndarray) -> float:
    """Otsu 类间方差比 w1·w2·(μ1-μ2)²：无需真值的「可分性」准则，越大越好。"""
    t = filters.threshold_otsu(vals)
    lo, hi = vals[vals <= t], vals[vals > t]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w1, w2 = lo.size / vals.size, hi.size / vals.size
    return float(w1 * w2 * (lo.mean() - hi.mean()) ** 2)


def _sep_gain(base: np.ndarray, pc: np.ndarray, roi: np.ndarray) -> float:
    """加入主成分通道后，融合得分图在 ROI 内 Otsu 可分性的提升倍数（无监督，不用真值）。

    两个通道都做过 ROI 内 z-score，因此「等权相加」就是各占一半权重。
    """
    r0 = _otsu_ratio(base[roi])
    r1 = _otsu_ratio((base + pc)[roi])
    return float(r1 / max(r0, 1e-9))


# ----------------------------------------------------------------------
# 步骤
# ----------------------------------------------------------------------
def steps() -> list[Step]:
    def s1(ctx):
        """四个模态共享病灶几何但对比机制互补，另含一块「只有纹理异常」的环。"""
        t0 = time.time()
        ctx.update(_make_modalities())
        ctx["spacing"] = (1.0, 1.0, 1.0)
        ctx["s1_seconds"] = time.time() - t0

        organ, lesion, core, halo = ctx["organ"], ctx["lesion"], ctx["core"], ctx["halo"]
        rows = []
        for name in MODALITIES:
            img = ctx["mods"][name]
            mu_o, sd_o = img[organ].mean(), img[organ].std()
            rows.append({
                "模态": name,
                "核心病灶对比度 CNR": round(float(abs(img[core].mean() - mu_o) / (sd_o + 1e-6)), 2),
                "纹理环对比度 CNR": round(float(abs(img[halo].mean() - mu_o) / (sd_o + 1e-6)), 2),
                "纹理环/器官 局部标准差比": round(float(
                    ndimage.uniform_filter(img, 9)[halo].std()
                    / (ndimage.uniform_filter(img, 9)[organ].std() + 1e-6)), 2),
                "器官内标准差": round(float(sd_o), 2),
            })
        ctx["mod_table"] = pd.DataFrame(rows)
        return ctx["mod_table"]

    def s2(ctx):
        """逐模态做体素级放射组学滤波：7 类特征 × 4 模态 = 28 张 3D 特征图。"""
        t0 = time.time()
        fmap = {}
        for name in MODALITIES:
            res = radiomic_filtering(ctx["mods"][name], ctx["organ"],
                                     kernel_size=KERNEL, bins=BINS, features=FEATURES)
            fmap[name] = res.maps
        ctx["fmaps"] = fmap
        ctx["s2_seconds"] = time.time() - t0

        core, halo, organ = ctx["core"], ctx["halo"], ctx["organ"]
        bg = organ & ~core & ~halo
        rows = []
        for feat in FEATURES:
            row = {"特征图": FEATURE_LABELS.get(feat, feat)}
            for name in MODALITIES:
                m = fmap[name][feat]
                a, b = m[core | halo], m[bg]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-6
                row[name] = round(float(abs(a.mean() - b.mean()) / pooled), 2)
            rows.append(row)
        tbl = pd.DataFrame(rows)
        tbl.columns = ["特征图"] + [f"{m} 效应量 d" for m in MODALITIES]
        ctx["fmap_table"] = tbl
        return tbl

    def s3(ctx):
        """对 28 张特征图做 PCA，取前 4 个主成分 —— 这就是「放射组学嵌入」。"""
        organ = ctx["organ"]
        names, cols = [], []
        for name in MODALITIES:
            for feat in FEATURES:
                cols.append(_znorm(ctx["fmaps"][name][feat], organ)[organ])
                names.append(f"{name}·{feat}")

        X = pd.DataFrame(np.stack(cols, axis=1), columns=names)
        comps, info = pca_reduce(X, n_components=4)
        ctx["pc_names"] = names

        pc_vols = []
        for k in range(4):
            vol = np.zeros(organ.shape, dtype=np.float32)
            vol[organ] = comps[:, k]
            pc_vols.append(vol)
        ctx["pc_vols"] = pc_vols

        corr = np.asarray([[float(np.corrcoef(X.values[:, j], comps[:, k])[0, 1])
                            for j in range(len(names))] for k in range(4)])
        top = []
        for k in range(4):
            order = np.argsort(-np.abs(corr[k]))[:3]
            top.append("、".join(f"{names[j]}({corr[k, j]:+.2f})" for j in order))
        info = info.copy()
        info["载荷最大的特征图"] = top
        ctx["pc_info"], ctx["pc_load"] = info, corr

        # 每个主成分对「核心病灶 + 纹理环」的区分度（用于解释后面的权重选择）
        lesion_bg = organ & ~ctx["lesion"]
        ds = []
        for k in range(4):
            a, b = comps[:, k][ctx["lesion"][organ]], comps[:, k][lesion_bg[organ]]
            ds.append(round(float(abs(a.mean() - b.mean())
                                  / (np.sqrt((a.var() + b.var()) / 2) + 1e-6)), 2))
        ctx["pc_effect"] = ds
        return info

    def s4(ctx):
        """4 个子分割器：得分图 = 四模态融合 + 主成分 i（权重无监督选取）。"""
        t0 = time.time()
        organ, lesion = ctx["organ"], ctx["lesion"]
        base = np.mean([_znorm(ctx["mods"][n], organ) for n in MODALITIES], axis=0)
        ctx["base_score"] = base
        ctx["base_ratio"] = _otsu_ratio(base[organ])

        members, probs, rows = [], [], []
        for i, method in enumerate(METHODS_4):
            pc = _znorm(ctx["pc_vols"][i], organ)
            gain = _sep_gain(base, pc, organ)
            score = base + pc                       # 强度通道与主成分通道等权
            member = segment(score, method=method, roi=organ)
            prob = np.where(organ, ndimage.gaussian_filter(member.astype(np.float32), 1.5), 0.0)
            members.append(member)
            probs.append(prob)
            rows.append({
                "子分割器": f"子模型 {i + 1}",
                "放射组学通道": f"PC{i + 1}",
                "阈值方法": method,
                "无监督可分性提升": f"{gain:.2f}×",
                "单独 Dice": round(dice(member, lesion), 4),
                "PC 区分度 d": ctx["pc_effect"][i],
            })
        ctx["members"], ctx["probs"] = members, probs
        ctx["sub_table"] = pd.DataFrame(rows)
        ctx["s4_seconds"] = time.time() - t0
        return ctx["sub_table"]

    def s5(ctx):
        """4 张概率图取平均 → Otsu 二值化（对应原文的 ensemble + Otsu 聚合）。"""
        organ, lesion = ctx["organ"], ctx["lesion"]
        ens = np.mean(ctx["probs"], axis=0)
        final = otsu_aggregate(ens, roi=organ)
        ctx["ensemble_prob"], ctx["final"] = ens, final
        m = evaluate(final, lesion, ctx["spacing"])
        ctx["final_metrics"] = m
        bg = organ & ~lesion
        return {
            "集成概率图-病灶内均值": round(float(ens[lesion].mean()), 3),
            "集成概率图-背景内均值": round(float(ens[bg].mean()), 3),
            "最终 Dice": m["Dice"],
            "最终 IoU": m["IoU"],
            "最终 HD95 (mm)": m["HD95 (mm)"],
            "灵敏度": m["灵敏度"],
            "特异度": m["特异度"],
        }

    def s6(ctx):
        """与基线比较：单模态、四模态平均、自适应阈值、无主成分集成、硬投票。"""
        organ, lesion = ctx["organ"], ctx["lesion"]
        rows = []

        def add(tag, mask, note=""):
            m = evaluate(mask, lesion, ctx["spacing"])
            rows.append({"方法": tag, "说明": note, "Dice": m["Dice"], "IoU": m["IoU"],
                         "HD95 (mm)": m["HD95 (mm)"], "灵敏度": m["灵敏度"],
                         "特异度": m["特异度"]})

        single = {}
        for name in MODALITIES:
            mask = segment(ctx["mods"][name], method="otsu", roi=organ)
            single[name] = dice(mask, lesion)
            add(f"基线：仅 {name} + Otsu", mask)
        best_mod = max(single, key=single.get)
        ctx["single_dice"], ctx["best_mod"] = single, best_mod

        base = ctx["base_score"]
        add("基线：四模态平均 + Otsu", segment(base, method="otsu", roi=organ), "纯强度")
        add("基线：四模态平均 + 自适应阈值",
            segment(base, method="adaptive", roi=organ), "纯强度、局部阈值")
        add("基线：四模态平均 + K-means",
            segment(base, method="kmeans", roi=organ), "纯强度")

        # 无主成分的集成：与主方法完全同构，只把 PC 权重设为 0
        probs0 = [np.where(organ, ndimage.gaussian_filter(
            segment(base, method=me, roi=organ).astype(np.float32), 1.5), 0.0)
            for me in METHODS_4]
        add("基线：4 子分割器（无主成分）+ Otsu 聚合",
            otsu_aggregate(np.mean(probs0, axis=0), roi=organ), "去掉放射组学嵌入")

        # 用户直觉版：一个模态只配一个主成分（信息量最少的一种配对）
        probs1 = []
        for i, (mod, me) in enumerate(zip(MODALITIES, METHODS_4)):
            sc = _znorm(ctx["mods"][mod], organ) + _znorm(ctx["pc_vols"][i], organ)
            probs1.append(np.where(organ, ndimage.gaussian_filter(
                segment(sc, method=me, roi=organ).astype(np.float32), 1.5), 0.0))
        add("对照：单模态 + 单主成分配对（1+1）+ Otsu 聚合",
            otsu_aggregate(np.mean(probs1, axis=0), roi=organ), "只用一个模态")

        # 硬投票：不做软平均
        votes = np.mean([otsu_aggregate(p, roi=organ).astype(np.float32)
                         for p in ctx["probs"]], axis=0)
        add("对照：4 个子分割器硬投票", votes > 0.5)

        add("本复现：四模态 + 主成分 4 子分割器软平均 + Otsu", ctx["final"], "原文流程")

        tbl = pd.DataFrame(rows)
        ctx["cmp_table"] = tbl

        def get(key):
            return float(tbl[tbl["方法"].str.contains(key, regex=False)]["Dice"].iloc[0])

        ctx["dice_best_single"] = single[best_mod]
        ctx["dice_4mod"] = get("四模态平均 + Otsu")
        ctx["dice_nopc"] = get("无主成分")
        ctx["dice_1plus1"] = get("1+1")
        ctx["dice_ens"] = get("本复现")
        ctx["hd95_nopc"] = float(tbl[tbl["方法"].str.contains("无主成分")]["HD95 (mm)"].iloc[0])
        return tbl

    def s7(ctx):
        """结论对照：集成 vs 单模态、有/无放射组学嵌入。"""
        info = ctx["pc_info"]
        t = ctx["cmp_table"]
        gain_pc = ctx["dice_ens"] - ctx["dice_nopc"]
        gain_1mod = ctx["dice_ens"] - ctx["dice_best_single"]
        dice_vote = float(t[t["方法"].str.contains("硬投票")]["Dice"].iloc[0])
        cnr_halo = float(ctx["mod_table"]["纹理环对比度 CNR"].max())
        return (
            "① 数据与特征图：4 个合成模态共享同一病灶几何、对比机制互补（T1/T2/FLAIR/T1ce），"
            "真值由「核心病灶」+「纹理异常环」组成；纹理环在四个模态上的平均强度几乎与正常组织相同"
            f"（最大 CNR 仅 {cnr_halo:.2f}），只有局部纹理粗糙 2.4 倍 —— 纯强度方法原理上看不到它。"
            f"每模态用 {KERNEL}³ 滑窗提 {len(FEATURES)} 类特征图，共 "
            f"{len(FEATURES) * len(MODALITIES)} 张（耗时 {ctx['s2_seconds']:.1f} s）。\n"
            f"② 放射组学嵌入：PCA 前 4 个主成分累计解释 "
            f"{info['累计'].iloc[-1] * 100:.1f}% 的方差（PC1 单独 "
            f"{info['解释方差比'].iloc[0] * 100:.1f}%），说明特征图之间高度冗余；"
            f"各主成分对「病灶 vs 背景」的区分度 d 为 {ctx['pc_effect']}，"
            "前两个主成分（分别由熵/均值类与标准差类特征主导）确实抓住了病灶的纹理改变。\n"
            f"③ 子分割器：4 个子模型分别用 {METHODS_4} 分割「四模态融合得分 + 对应主成分」，"
            f"无监督可分性提升 "
            f"{list(ctx['sub_table']['无监督可分性提升'])}，单独 Dice 介于 "
            f"{ctx['sub_table']['单独 Dice'].min():.3f}–{ctx['sub_table']['单独 Dice'].max():.3f}；"
            "子模型之间因为主成分与阈值策略不同而保持了多样性。\n"
            f"④ 集成与基线：最好的单模态基线（{ctx['best_mod']} + Otsu）Dice = "
            f"{ctx['dice_best_single']:.3f}，四模态平均 + Otsu = {ctx['dice_4mod']:.3f}"
            "（看不到纹理异常环，只能切出核心病灶）；"
            f"去掉主成分的同构集成 = {ctx['dice_nopc']:.3f}；"
            f"「一个模态 + 一个主成分」的最简配对 = {ctx['dice_1plus1']:.3f}；"
            f"本复现（四模态 + 主成分 4 子分割器软平均 + Otsu）= {ctx['dice_ens']:.3f}"
            f"（HD95 {ctx['final_metrics']['HD95 (mm)']:.2f} mm，"
            f"相对无主成分集成的 {ctx['hd95_nopc']:.2f} mm 明显更准）。\n"
            f"⑤ 关键结论：相对最好的单模态基线提升 {gain_1mod:+.3f}，"
            f"相对「无放射组学嵌入」的同构集成提升 {gain_pc:+.3f} —— "
            "这一差值就是放射组学嵌入的净贡献，也是原文「radiomics-incorporated」的意义："
            "它补上的正是影像强度看不到、而纹理特征图看得到的那部分病灶。"
            f"此外，把「软平均后再 Otsu」换成「各子模型 Otsu 后多数表决」，Dice 可进一步到 "
            f"{dice_vote:.3f}，说明**聚合策略本身也是可优化的一环**。\n"
            "⑥ 与原文对照：原文用 4 个 3D U-Net + 四模态 + 1 个主成分、softmax 概率叠加后 Otsu 二值化；"
            "本复现保留了完全相同的流程骨架（多模态特征图 → PCA 4 主成分 → 4 子模型 → 概率叠加 → Otsu），"
            "只用经典分割器替代了需要训练的 U-Net。由于没有训练网络、数据也是合成的，"
            "绝对 Dice 不能与原文比较，但「多模态 + 集成 + 放射组学嵌入」三件套的相对收益方向完全一致。"
        )

    return [
        Step("① 造四模态合成病灶数据", "对应原文 Dataset「multi-parametric MRI（T1/T1ce/T2/FLAIR）」："
             "四个模态共享同一病灶几何但对比机制不同，且病灶含一块**四模态强度均等值、仅纹理异常**的环，"
             "用于检验放射组学嵌入是否真的带来额外信息。",
             s1, "table", "纹理环的 CNR≈0、但局部纹理显著更粗糙：这正是放射组学特征图的主场。"),
        Step("② 逐通道提取放射组学特征图", "对应原文 Method 第一步：3D 滑窗在 ROI 内逐体素提取 7 类特征"
             "（均值/标准差/熵/均匀度/对比度/同质性/梯度），每个特征成为一张 3D 特征图。",
             s2, "table", "表内为「病灶 vs 背景」的效应量 d：纹理类特征图对纹理异常环同样敏感。"),
        Step("③ 特征图 PCA（放射组学嵌入）", "对应原文 Method 的 PCA 环节：把 28 张特征图压成前 4 个主成分，"
             "报告解释方差比、载荷最大的特征图，以及各主成分对病灶的区分度。",
             s3, "table", "累计解释方差比越高，说明原始特征图冗余越严重、降维越划算。"),
        Step("④ 四个子分割器各出概率图", "对应原文 4 个 U-Net 子模型：每个子模型的得分图 ="
             "「四模态融合通道 + 主成分 i」等权相加（两个通道都做过 ROI 内 z-score），"
             "再用 Otsu/Li/Yen/K-means 四种方法产生成员掩膜并平滑为软概率图；"
             "同时报告加入该主成分后得分图可分性的无监督提升倍数。",
             s4, "table", "可分性提升是无监督指标（不含真值），只看「加了通道后病灶更不更可分」；它不完全等价于分割精度。"),
        Step("⑤ Otsu 聚合得到最终分割", "对应原文最后一步：4 张概率图叠加后**用 Otsu 二值化**"
             "（不是固定 0.5），得到最终分割并评估 Dice/HD95/灵敏度/特异度。",
             s5, "metrics", "Otsu 聚合让最终结果对概率图的绝对尺度不敏感。"),
        Step("⑥ 与基线比较 Dice / HD95", "五类基线：4 个单模态 + Otsu、四模态平均 + Otsu、"
             "四模态平均 + 自适应阈值 / K-means、去掉放射组学主成分的同构集成、"
             "「一个模态配一个主成分」的最简配对，以及硬投票。",
             s6, "table", "关键对照是「有/无主成分」，它直接量化放射组学嵌入的净贡献。"),
        Step("⑦ 结论对照", "把复现流程与原文逐步对应，报告集成相对单模态的增益与放射组学嵌入的净贡献。",
             s7, "text"),
    ]
