"""
特征建模与预后预测（Feature Modeling）
========================================
对应文献：
  - Yang Z, et al. Development of a multi-feature-combined model ... local failure
    prediction of post-SBRT or surgery early-stage NSCLC patients. Front Oncol 2023;13:1185771.
    （MFC 模型：手工放射组学 + 深度特征 + 临床信息三源融合）
  - Zhang R, ..., Yang Z. A dual-radiomics model for overall survival prediction in
    early-stage NSCLC. Front Oncol 2024;14:1419621.（双放射组学）
  - Hu Z, Yang Z, et al. A Deep Learning Model with Radiomics Analysis Integration
    for Glioblastoma Post-Resection Survival Prediction. arXiv:2203.05891.

覆盖的方法学要素（与原论文一一对应）：
  多共线性评估 · PCA 降维 · 差异性分析特征选择 · LR/SVM/RF 分类器 ·
  LOOCV / k 折 / 蒙特卡洛交叉验证 · ROC-AUC / 灵敏度 / 特异度 · 置换重要性
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

FEATURE_SOURCES = ["手工放射组学", "深度特征", "临床信息"]


# ----------------------------------------------------------------------
# 合成队列：模拟「三源特征 + 二分类结局」
# ----------------------------------------------------------------------
@dataclass
class Cohort:
    X: pd.DataFrame
    y: np.ndarray
    sources: dict[str, list[str]]
    risk_score: np.ndarray


def make_synthetic_cohort(n: int = 160, seed: int = 0,
                          n_radiomic: int = 8, n_deep: int = 6, n_clinical: int = 4,
                          signal: float = 1.0) -> Cohort:
    """生成一个「三源特征」的合成队列。

    设计：结局由三源共同决定，但**单一来源的信息都不完整**——
    这正是 MFC 论文的出发点（任何单一来源都不够，融合才更好）。
      - 手工放射组学：与结局中等相关
      - 深度特征：与结局弱相关，但与放射组学部分冗余
      - 临床信息：与结局弱相关，且与影像特征独立
    """
    rng = np.random.RandomState(seed)

    latent = rng.randn(n)                                   # 潜在风险因子
    radio = np.column_stack([latent * 0.75 + rng.randn(n) * 0.65 for _ in range(n_radiomic)])
    deep = np.column_stack([latent * 0.45 + rng.randn(n) * 0.90 for _ in range(n_deep)])
    clinical = np.column_stack([
        latent * 0.25 + rng.randn(n) * 0.95,               # 年龄
        latent * 0.20 + rng.randn(n) * 0.95,               # 肿瘤体积
        rng.randn(n),                                      # 性别（与结局无关）
        latent * 0.30 + rng.randn(n) * 0.95,               # 合并症指数
    ])

    def name(prefix, k):
        return [f"{prefix}_{i+1:02d}" for i in range(k)]

    sources = {
        "手工放射组学": name("radiomic", n_radiomic),
        "深度特征": name("deep", n_deep),
        "临床信息": ["age", "tumor_volume", "gender", "CCI"],
    }

    X = pd.DataFrame(np.column_stack([radio, deep, clinical]),
                     columns=sources["手工放射组学"] + sources["深度特征"] + sources["临床信息"])

    logit = signal * (latent * 1.25 - 0.3)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.rand(n) < p).astype(int)

    return Cohort(X=X, y=y, sources=sources, risk_score=latent)


# ----------------------------------------------------------------------
# 多共线性评估（对应论文里的 multi-collinearity assessment）
# ----------------------------------------------------------------------
def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr()


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """方差膨胀因子：VIF > 10 通常认为存在严重共线性。"""
    rows = []
    cols = list(X.columns)
    for c in cols:
        others = [o for o in cols if o != c]
        y = X[c].values
        A = np.column_stack([np.ones(len(X))] + [X[o].values for o in others])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / max(1 - r2, 1e-9)
        except Exception:
            vif = float("nan")
        rows.append({"特征": c, "VIF": round(float(vif), 3),
                     "判断": "⚠️ 严重共线" if vif > 10 else ("注意" if vif > 5 else "可接受")})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def prune_collinear(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """剔除高度相关（|r| > threshold）的冗余特征，保留先出现的那个。"""
    corr = X.corr().abs()
    keep: list[str] = []
    for c in X.columns:
        if all(corr.loc[c, k] <= threshold for k in keep):
            keep.append(c)
    return keep


# ----------------------------------------------------------------------
# 降维与特征选择
# ----------------------------------------------------------------------
def pca_reduce(X: pd.DataFrame, n_components: int = 4) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(Z)
    info = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(n_components)],
        "解释方差比": np.round(pca.explained_variance_ratio_, 4),
        "累计": np.round(np.cumsum(pca.explained_variance_ratio_), 4),
    })
    return comps, info


def dissimilarity_select(X: pd.DataFrame, y: np.ndarray, k: int = 6) -> list[str]:
    """差异性分析：挑出「组间差异最大」的特征（对应 GBM 论文中的做法）。"""
    scores = {}
    for c in X.columns:
        a, b = X[c].values[y == 1], X[c].values[y == 0]
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
        scores[c] = abs(a.mean() - b.mean()) / pooled      # Cohen's d
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


# ----------------------------------------------------------------------
# 交叉验证与模型评估
# ----------------------------------------------------------------------
def _make_model(name: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    if name == "SVM":
        return make_pipeline(StandardScaler(),
                             SVC(probability=True, kernel="rbf", C=1.0, random_state=0))
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=0,
                                      min_samples_leaf=3)
    raise ValueError(name)


def cv_predict(name: str, X: np.ndarray, y: np.ndarray,
               scheme: str = "kfold", n_splits: int = 5,
               n_repeats: int = 50, test_size: float = 0.3,
               seed: int = 0) -> dict:
    """交叉验证并返回样本外预测概率。

    scheme：
      - "kfold"  ：k 折交叉验证
      - "loocv"  ：留一法
      - "mccv"   ：蒙特卡洛交叉验证（重复随机划分，原论文用 100 次 7:3）
    """
    from sklearn.base import clone
    from sklearn.model_selection import (KFold, LeaveOneOut, StratifiedKFold,
                                         train_test_split)

    rng = np.random.RandomState(seed)
    oof = np.full(len(y), np.nan)

    if scheme == "loocv":
        splitter = LeaveOneOut()
        for tr, te in splitter.split(X, y):
            m = _make_model(name).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    elif scheme == "kfold":
        k = min(n_splits, int(min(np.bincount(y))))
        splitter = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=seed)
        for tr, te in splitter.split(X, y):
            m = _make_model(name).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    elif scheme == "mccv":
        acc = np.zeros(len(y))
        cnt = np.zeros(len(y))
        for i in range(n_repeats):
            tr, te = train_test_split(np.arange(len(y)), test_size=test_size,
                                      random_state=rng.randint(1 << 30), stratify=y)
            m = _make_model(name).fit(X[tr], y[tr])
            acc[te] += m.predict_proba(X[te])[:, 1]
            cnt[te] += 1
        cnt[cnt == 0] = 1
        oof = acc / cnt
    else:
        raise ValueError(scheme)

    return {"y_true": y, "y_score": oof, "scheme": scheme,
            "metrics": binary_metrics(y, oof)}


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   threshold: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 roc_auc_score)

    ok = np.isfinite(y_score)
    yt, ys = y_true[ok], y_score[ok]
    pred = (ys >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {
        "AUC": round(float(roc_auc_score(yt, ys)), 4) if len(set(yt)) > 1 else float("nan"),
        "准确率": round(float(accuracy_score(yt, pred)), 4),
        "F1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "灵敏度": round(float(tp / max(tp + fn, 1)), 4),
        "特异度": round(float(tn / max(tn + fp, 1)), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve
    ok = np.isfinite(y_score)
    fpr, tpr, _ = roc_curve(y_true[ok], y_score[ok])
    return fpr, tpr


# ----------------------------------------------------------------------
# 可解释性：置换重要性（对应论文里的特征贡献分析 / LRP 的替代品）
# ----------------------------------------------------------------------
def permutation_importance_table(name: str, X: pd.DataFrame, y: np.ndarray,
                                 n_repeats: int = 12, seed: int = 0) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    m = _make_model(name).fit(Xtr, ytr)
    r = permutation_importance(m, Xte, yte, n_repeats=n_repeats,
                               random_state=seed, scoring="roc_auc")
    df = pd.DataFrame({
        "特征": X.columns,
        "AUC 下降（均值）": np.round(r.importances_mean, 4),
        "标准差": np.round(r.importances_std, 4),
    })
    return df.sort_values("AUC 下降（均值）", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 三源融合对比（复现 MFC 论文的核心结论）
# ----------------------------------------------------------------------
def fusion_experiment(cohort: Cohort, model: str = "LR",
                      scheme: str = "kfold", n_repeats: int = 30) -> pd.DataFrame:
    """比较「单一来源」与「多源融合」的判别能力。"""
    sources = cohort.sources
    combos = []
    for r in range(1, len(sources) + 1):
        for c in combinations(sources.keys(), r):
            combos.append(c)

    rows = []
    for combo in combos:
        cols = [c for s in combo for c in sources[s]]
        res = cv_predict(model, cohort.X[cols].values, cohort.y,
                         scheme=scheme, n_repeats=n_repeats)
        rows.append({
            "特征来源": " + ".join(combo),
            "特征数": len(cols),
            **{k: v for k, v in res["metrics"].items()
               if k in ("AUC", "准确率", "灵敏度", "特异度")},
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
