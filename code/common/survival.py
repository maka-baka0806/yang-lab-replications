"""
生存分析与风险分层（Survival Analysis）
=========================================
对应文献：
  - Zhang R, ..., Yang Z. A dual-radiomics model for overall survival prediction in
    early-stage NSCLC patient using pre-treatment CT images. Front Oncol 2024;14:1419621.

论文的核心动作：
  1. 用放射组学特征算出一个「风险分数」
  2. 按中位数把患者分成 高/低风险 两组
  3. 画 Kaplan-Meier 曲线，用 log-rank 检验比较两组生存差异

本模块手工实现 KM 估计与 log-rank 检验（不依赖第三方生存分析库），
这样每个数字都能追溯到公式。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 合成生存数据
# ----------------------------------------------------------------------
@dataclass
class SurvivalData:
    time: np.ndarray        # 随访时间（月）
    event: np.ndarray       # 1 = 死亡/事件，0 = 删失
    risk: np.ndarray        # 风险分数（来自放射组学模型）
    group: np.ndarray       # 0 = 低风险，1 = 高风险


def simulate_survival(risk: np.ndarray, base_hazard: float = 0.08,
                      effect: float = 1.5, max_follow: float = 60.0,
                      censor_rate: float = 0.25, seed: int = 0) -> SurvivalData:
    """由风险分数生成生存时间（指数模型 + 随机删失）。

    hazard_i = base_hazard * exp(effect * z_i)，其中 z 是标准化后的风险分数。
    风险越高 → 风险率越高 → 生存时间越短。
    """
    rng = np.random.RandomState(seed)
    z = (risk - risk.mean()) / (risk.std() + 1e-9)
    hazard = base_hazard * np.exp(effect * z)
    t_event = rng.exponential(1.0 / hazard)

    # 随机删失（失访）
    t_censor = rng.exponential(max_follow / max(censor_rate * 4, 0.5), size=len(risk))
    time = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(int)

    # 行政删失（随访截止）
    time = np.minimum(time, max_follow)
    group = (z >= 0).astype(int)

    return SurvivalData(time=time, event=event, risk=z, group=group)


# ----------------------------------------------------------------------
# Kaplan-Meier 估计
# ----------------------------------------------------------------------
def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """返回 KM 阶梯曲线的 (时间, 生存概率, 置信区间下界, 上界)。

    生存函数 S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)
    方差用 Greenwood 公式，置信区间用 log-log 变换（与主流软件一致）。
    """
    order = np.argsort(time)
    t, e = np.asarray(time)[order], np.asarray(event)[order]
    uniq = np.unique(t)

    n_at_risk, surv = len(t), 1.0
    greenwood = 0.0
    rows = [{"时间": 0.0, "生存概率": 1.0, "n_at_risk": n_at_risk,
             "下界": 1.0, "上界": 1.0}]

    for ti in uniq:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        if at_risk == 0:
            continue
        if d > 0:
            surv *= (1 - d / at_risk)
            greenwood += d / (at_risk * (at_risk - d)) if at_risk > d else 0.0
        se = surv * np.sqrt(greenwood) if greenwood > 0 else 0.0
        if 0 < surv < 1 and se > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ll = np.log(-np.log(surv))
                se_ll = se / (surv * abs(np.log(surv)))
            lo = surv ** np.exp(1.96 * se_ll)
            hi = surv ** np.exp(-1.96 * se_ll)
        else:
            lo, hi = surv, surv
        rows.append({"时间": float(ti), "生存概率": float(surv),
                     "n_at_risk": at_risk, "下界": float(lo), "上界": float(hi)})

    return pd.DataFrame(rows)


def median_survival(km: pd.DataFrame) -> float:
    """中位生存时间：生存概率首次降到 0.5 以下的时间。"""
    below = km[km["生存概率"] <= 0.5]
    return float(below["时间"].iloc[0]) if len(below) else float("nan")


# ----------------------------------------------------------------------
# log-rank 检验
# ----------------------------------------------------------------------
def logrank_test(time: np.ndarray, event: np.ndarray,
                 group: np.ndarray) -> dict:
    """两组 log-rank 检验（Mantel-Cox）。

    在每个事件时点计算观测数 O 与期望数 E，构造卡方统计量：
        χ² = (O1 - E1)² / V，  V 为超几何方差
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group, int)
    g1 = group == 1

    O1_minus_E1 = 0.0
    V = 0.0
    for ti in np.unique(time[event == 1]):
        at_risk = time >= ti
        n = int(at_risk.sum())
        n1 = int((at_risk & g1).sum())
        d = int(((time == ti) & (event == 1)).sum())
        d1 = int(((time == ti) & (event == 1) & g1).sum())
        if n <= 1:
            continue
        e1 = d * n1 / n
        v = (n1 * (n - n1) * d * (n - d)) / (n * n * (n - 1))
        O1_minus_E1 += d1 - e1
        V += v

    if V <= 0:
        return {"chi2": float("nan"), "p": float("nan"), "df": 1}
    chi2 = (O1_minus_E1 ** 2) / V
    # 卡方分布（1 自由度）的生存函数：p = erfc(sqrt(chi2/2))
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2.0))
    return {"chi2": round(float(chi2), 4), "p": round(float(p), 6), "df": 1}


def stratified_analysis(data: SurvivalData) -> dict:
    """按风险分数中位数分成高/低风险两组并做完整分析。"""
    out = {}
    for g, label in [(0, "低风险"), (1, "高风险")]:
        sel = data.group == g
        km = kaplan_meier(data.time[sel], data.event[sel])
        out[label] = {
            "n": int(sel.sum()),
            "事件数": int(data.event[sel].sum()),
            "KM": km,
            "中位生存（月）": median_survival(km),
        }
    out["logrank"] = logrank_test(data.time, data.event, data.group)
    return out


def risk_stratification_table(risk: np.ndarray, time: np.ndarray,
                              event: np.ndarray) -> pd.DataFrame:
    """按风险三分位做分层（更细的风险分层，常用于论文中的补充分析）。"""
    q = np.quantile(risk, [0, 1 / 3, 2 / 3, 1.0])
    labels = ["低风险", "中风险", "高风险"]
    rows = []
    for i in range(3):
        sel = (risk >= q[i]) & (risk <= q[i + 1] if i == 2 else risk < q[i + 1])
        if sel.sum() == 0:
            continue
        rows.append({
            "分层": labels[i],
            "n": int(sel.sum()),
            "事件数": int(event[sel].sum()),
            "中位生存（月）": round(median_survival(kaplan_meier(time[sel], event[sel])), 1),
        })
    return pd.DataFrame(rows)
