from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np


# ---------------------------
# Utilities / small helpers
# ---------------------------

def _to_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)

def _symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)

def _psd_project(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project symmetric matrix to PSD by clipping eigenvalues."""
    A = _symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return V @ np.diag(w) @ V.T

def _safe_inv(A: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    A = _symmetrize(A)
    return np.linalg.inv(A + ridge * np.eye(A.shape[0]))


# ===========================
# Chapter 1 — Introduction
# ===========================

def ch01_active_management_value_chain(
    forecast_alpha: np.ndarray,
    risk_model: np.ndarray,
    constraints: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    transaction_cost: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, object]:
    """
    Chapter 1 核心: 主动管理是“价值链系统”, 不是单点技巧。
    书的全局结构基本就是: 预期收益(Alpha) + 风险(Σ) + 组合构建(优化/约束)
    + 交易实现(成本/冲击/换手) + 绩效评估(归因/检验)。

    Parameters
    ----------
    forecast_alpha : (N,) alpha/预期超额收益向量
    risk_model     : (N,N) 协方差矩阵 Σ(或者风险模型还原的 Σ)
    constraints    : 可选, 把原始最优权重投影/修正到可行域
    transaction_cost : 可选, 给定“交易量/换仓”返回成本

    Returns
    -------
    dict: 系统组件的“接口契约”提示, 用于后续章节函数的衔接。

    Example
    -------
    >>> N=3
    >>> alpha=np.array([0.02, 0.01, -0.005])
    >>> Sigma=np.eye(N)*0.04
    >>> ch01_active_management_value_chain(alpha, Sigma)["stage_order"]
    ('forecast', 'risk', 'optimize', 'trade', 'evaluate')
    """
    return {
        "alpha": np.asarray(forecast_alpha, float),
        "Sigma": _psd_project(risk_model),
        "constraints": constraints,
        "transaction_cost": transaction_cost,
        "stage_order": ("forecast", "risk", "optimize", "trade", "evaluate"),
    }


# ==========================================
# Chapter 2 — CAPM (Consensus Expected Return)
# ==========================================

def ch02_capm_expected_excess_return(
    beta: np.ndarray,
    market_expected_excess_return: float,
) -> np.ndarray:
    r"""
    Chapter 2 核心: CAPM 给出“共识”预期收益(equilibrium / consensus)。
    在 APM 体系里, 它常用作: 基准/先验/或“你不做主动时该拿到的那部分”。

    公式(超额收益形式): 
        E[r_i] = beta_i * E[r_m]

    Parameters
    ----------
    beta : (N,) 资产对市场的 β
    market_expected_excess_return : 市场长期预期超额收益 E[r_m]

    Returns
    -------
    (N,) 共识预期超额收益向量

    Example
    -------
    >>> betas=np.array([1.1, 0.8])
    >>> ch02_capm_expected_excess_return(betas, 0.06)
    array([0.066, 0.048])
    """
    beta = np.asarray(beta, float)
    return beta * float(market_expected_excess_return)


# ======================
# Chapter 3 — Risk
# ======================

def ch03_factor_risk_model(
    B: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    r"""
    Chapter 3 核心: 结构化风险模型(factor model)把 Σ 分解成可估计、可解释的块。

        Σ = B F B' + D

    - B: (N,K) 因子暴露/载荷
    - F: (K,K) 因子协方差
    - D: (N,N) 特质风险(常为对角)

    这一步是 APM 的工程“底盘”: N 很大时, 直接估 Σ 不稳, factor 化更稳健。

    Example
    -------
    >>> N,K=4,2
    >>> B=np.random.randn(N,K)
    >>> F=np.eye(K)*0.02
    >>> D=np.eye(N)*0.03
    >>> Sigma=ch03_factor_risk_model(B,F,D)
    >>> Sigma.shape
    (4, 4)
    """
    B = np.asarray(B, float)
    F = _psd_project(F)
    D = np.asarray(D, float)
    Sigma = B @ F @ B.T + D
    return _psd_project(Sigma)


# ======================================================
# Chapter 4 — Exceptional Return, Benchmarks, Value Added
# ======================================================

def ch04_value_added(
    portfolio_return: float,
    benchmark_return: float,
) -> float:
    r"""
    Chapter 4 核心: 主动收益(value added / active return)= 组合相对基准的超额。

        VA = R_p - R_b

    这看似简单, 但它定义了主动管理的“计分方式”, 也决定了后续风险/优化目标。

    Example
    -------
    >>> ch04_value_added(0.012, 0.010)
    0.002
    """
    return float(portfolio_return) - float(benchmark_return)


# =================================================
# Chapter 5 — Residual Risk/Return: Information Ratio
# =================================================

def ch05_information_ratio(
    active_returns: np.ndarray,
    annualization_factor: float = 252.0,
) -> float:
    r"""
    Chapter 5 核心: IR 是主动管理的“单位风险收益”。

        IR = mean(active) / std(active)

    annualization_factor 用于把日频等缩放到年化(近似 √T 规则)。
    若 active_returns 是日度主动收益: 
        IR_ann ≈ (mean_d / std_d) * sqrt(252)

    Example
    -------
    >>> ar=np.array([0.001, -0.0005, 0.0008, 0.0002])
    >>> ch05_information_ratio(ar, 252.0)
    # 返回一个年化 IR(示意)
    """
    a = np.asarray(active_returns, float)
    mu = np.mean(a)
    sd = np.std(a, ddof=1) if a.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    # convert sample IR to annualized IR approximately
    ir = mu / sd
    return float(ir * np.sqrt(float(annualization_factor)))


# ===================================================
# Chapter 6 — Fundamental Law of Active Management
# ===================================================

def ch06_fundamental_law_ir(
    information_coefficient: float,
    breadth: float,
    transfer_coefficient: float = 1.0,
) -> float:
    r"""
    Chapter 6 核心: Fundamental Law(最出名的一条)。
    “你能做到的 IR”, 由三件事决定: 
    - IC: 预测与实现之间的相关(信息质量)
    - BR(breadth): 有效独立下注次数(信息广度)
    - TC: 从“理想信号”到“可实现组合”的转化效率(约束/成本/实现损耗)

    常见表达(不同版本写法略有差异, 这里用最常用形态): 
        IR ≈ TC * IC * sqrt(BR)

    Example
    -------
    >>> ch06_fundamental_law_ir(0.05, 400, 0.6)
    0.6*0.05*sqrt(400)=0.6
    """
    IC = float(information_coefficient)
    BR = float(breadth)
    TC = float(transfer_coefficient)
    return float(TC * IC * np.sqrt(max(BR, 0.0)))


# ==================================================
# Chapter 7 — Expected Returns and the APT
# ==================================================

def ch07_apt_expected_return(
    factor_exposures: np.ndarray,
    factor_risk_premia: np.ndarray,
    alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Chapter 7 核心: APT/多因子期望收益模型(“风险溢价 * 暴露”)。
    CAPM 是单因子; APT 是多因子的一般化。

        E[r] = alpha + B * lambda

    - B: (N,K)
    - lambda: (K,) 因子风险溢价
    - alpha: (N,) 可选的“非价格化/异常收益”(若你要把它放进收益模型里)

    Example
    -------
    >>> B=np.array([[1.0, 0.2],[0.8, -0.1]])
    >>> lam=np.array([0.05, 0.02])
    >>> ch07_apt_expected_return(B, lam)
    array([0.054, 0.038])
    """
    B = np.asarray(factor_exposures, float)
    lam = _to_col(np.asarray(factor_risk_premia, float))
    mu = (B @ lam).reshape(-1)
    if alpha is not None:
        mu = mu + np.asarray(alpha, float)
    return mu


# =================================
# Chapter 8 — Valuation in Theory
# =================================

def ch08_present_value(
    cashflows: np.ndarray,
    discount_rates: np.ndarray,
) -> float:
    r"""
    Chapter 8 核心: 估值理论 = 把未来现金流折现到今天。
    这章在 APM 里扮演“把基本面信息转成回报预期”的桥梁。

    最简单离散形式: 
        PV = sum_t CF_t / Π_{j<=t} (1 + r_j)

    Parameters
    ----------
    cashflows : (T,) 未来每期现金流
    discount_rates : (T,) 每期折现率(可变)

    Example
    -------
    >>> CF=np.array([1.0, 1.0, 101.0])
    >>> r =np.array([0.02,0.02,0.02])
    >>> round(ch08_present_value(CF,r), 2)
    97.13
    """
    CF = np.asarray(cashflows, float)
    r = np.asarray(discount_rates, float)
    assert CF.shape == r.shape
    df = np.cumprod(1.0 + r)
    return float(np.sum(CF / df))


# ===================================
# Chapter 9 — Valuation in Practice
# ===================================

def ch09_expected_return_from_valuation_gap(
    price: np.ndarray,
    intrinsic_value: np.ndarray,
    horizon: float,
    mean_reversion_speed: float = 1.0,
) -> np.ndarray:
    r"""
    Chapter 9 核心: 估值“落地”往往变成一个 alpha: 价格相对内在价值的偏离, 预期在某个期限内收敛。

    一个常用的可操作形式(示意): 
        gap = log(V / P)
        E[r_{0->H}] ≈ kappa * gap / H

    - horizon: H(以年计或你定义的单位)
    - mean_reversion_speed: κ(偏离收敛速度)

    Example
    -------
    >>> P=np.array([100, 50.0])
    >>> V=np.array([110, 45.0])
    >>> ch09_expected_return_from_valuation_gap(P,V,horizon=1.0,mean_reversion_speed=0.5)
    """
    P = np.asarray(price, float)
    V = np.asarray(intrinsic_value, float)
    H = float(horizon)
    kappa = float(mean_reversion_speed)
    gap = np.log(np.maximum(V, 1e-12) / np.maximum(P, 1e-12))
    return (kappa * gap / max(H, 1e-12)).astype(float)


# =================================
# Chapter 10 — Forecasting Basics
# =================================

def ch10_linear_forecast(
    features: np.ndarray,
    weights: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    r"""
    Chapter 10 核心: 预测 = 把信息映射成 alpha(最基础可用形态: 线性打分)。

        alpha_hat = intercept + X w

    这章更强调“预测对象、信息结构、噪声、可检验性”的基本功。
    我们用线性形式做一个最小可执行抽象。

    Example
    -------
    >>> X=np.array([[1,0],[0,1],[1,1]])
    >>> w=np.array([0.02,-0.01])
    >>> ch10_linear_forecast(X,w)
    array([ 0.02, -0.01,  0.01])
    """
    X = np.asarray(features, float)
    w = np.asarray(weights, float)
    return (float(intercept) + X @ w).astype(float)


# ===================================
# Chapter 11 — Advanced Forecasting
# ===================================

def ch11_information_coefficient(
    forecast: np.ndarray,
    realized: np.ndarray,
    method: str = "pearson",
) -> float:
    r"""
    Chapter 11 核心: IC(Information Coefficient)把“信号质量”变成可统计检验的量。
    常用: 截面相关(cross-sectional correlation)或时间序列相关, 取决于你的信号定义。

    Parameters
    ----------
    forecast : (N,) 或 (T,) 预测值
    realized : 同形状的实现收益/目标
    method : "pearson"(默认)。你也可以扩展 spearman/robust 等

    Example
    -------
    >>> f=np.array([0.1, 0.2, -0.1, 0.0])
    >>> r=np.array([0.05,0.03,-0.02,0.01])
    >>> ch11_information_coefficient(f,r)
    """
    f = np.asarray(forecast, float).reshape(-1)
    y = np.asarray(realized, float).reshape(-1)
    if f.size < 2:
        return 0.0
    if method != "pearson":
        raise ValueError("Only 'pearson' implemented in this minimal sketch.")
    f = f - f.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(f) * np.linalg.norm(y))
    return float((f @ y) / denom) if denom > 0 else 0.0


# ==================================
# Chapter 12 — Information Analysis
# ==================================

def ch12_ic_to_expected_ir(
    ic: float,
    breadth: float,
    transfer_coefficient: float = 1.0,
) -> float:
    r"""
    Chapter 12 核心: 信息分析把“信号统计量”翻译成“可期待的投资绩效”。
    这一章本质上就是: 如何严谨评估信息、避免自欺、并把 IC/预测误差映射到策略层面。

    最小抽象: 直接调用 Fundamental Law, 把“信息测量”连到“IR 预期”。

    Example
    -------
    >>> ch12_ic_to_expected_ir(0.03, 1000, 0.4)
    """
    return ch06_fundamental_law_ir(ic, breadth, transfer_coefficient)


# ===================================
# Chapter 13 — The Information Horizon
# ===================================

def ch13_ic_decay(
    ic0: float,
    half_life: float,
    dt: float,
) -> float:
    r"""
    Chapter 13 核心: 信息会衰减(horizon/half-life), 决定了持有期、换手与可实现 breadth。

    一个常用参数化: 指数衰减
        IC(t) = IC0 * exp(-lambda t)
    其中 half-life 满足: IC(half_life) = IC0/2
        lambda = ln(2)/half_life

    Example
    -------
    >>> round(ch13_ic_decay(0.05, half_life=10, dt=10), 5)
    0.025
    """
    ic0 = float(ic0)
    hl = float(half_life)
    dt = float(dt)
    lam = np.log(2.0) / max(hl, 1e-12)
    return float(ic0 * np.exp(-lam * dt))


# ==================================
# Chapter 14 — Portfolio Construction
# ==================================

def ch14_mean_variance_active_optimal_weights(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    risk_aversion: float = 1.0,
    w_b: Optional[np.ndarray] = None,
    enforce_dollar_neutral_active: bool = True,
) -> np.ndarray:
    r"""
    Chapter 14 核心: 组合构建(portfolio construction)把 alpha 与 Σ 转成权重。
    最经典的无约束二次优化(active space): 

        maximize   a' w_a - (λ/2) w_a' Σ w_a

    解为: 
        w_a* = (1/λ) Σ^{-1} a

    然后: 
        w = w_b + w_a

    Parameters
    ----------
    alpha : (N,) 预测的主动收益(相对基准)
    Sigma : (N,N) 风险(主动或总风险都行, 但要与你的 alpha 定义一致)
    risk_aversion : λ, 越大越保守
    w_b : (N,) 基准权重; None 则视为 0(纯主动)
    enforce_dollar_neutral_active : 若 True, 对 w_a 去均值, 保证 sum(w_a)=0(常见 active 管理假设)

    Example
    -------
    >>> N=3
    >>> a=np.array([0.02,0.01,-0.01])
    >>> S=np.eye(N)*0.04
    >>> w=ch14_mean_variance_active_optimal_weights(a,S,risk_aversion=2.0)
    """
    a = _to_col(np.asarray(alpha, float))
    S = _psd_project(Sigma)
    lam = float(risk_aversion)
    w_a = (1.0 / max(lam, 1e-12)) * (_safe_inv(S) @ a)
    w_a = w_a.reshape(-1)
    if enforce_dollar_neutral_active:
        w_a = w_a - w_a.mean()
    if w_b is None:
        return w_a
    return np.asarray(w_b, float) + w_a


# =================================
# Chapter 15 — Long/Short Investing
# =================================

def ch15_long_short_from_scores(
    scores: np.ndarray,
    gross_leverage: float = 2.0,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
) -> np.ndarray:
    r"""
    Chapter 15 核心: Long/Short 让你把信息更“纯粹”地表达在 active book 上, 
    但也把融资、借券、极端风险与交易成本放大了。

    一个最小可执行构造: 
    - top 分位做多, bottom 分位做空
    - 多头权重和空头权重分别归一化, 使得总 gross = gross_leverage

    Example
    -------
    >>> s=np.array([1.2, 0.1, -0.5, 2.0, -1.0])
    >>> w=ch15_long_short_from_scores(s, gross_leverage=2.0, top_frac=0.2, bottom_frac=0.2)
    >>> round(w.sum(), 6)  # 近似 0(市场中性)
    0.0
    """
    s = np.asarray(scores, float).reshape(-1)
    N = s.size
    k_top = max(1, int(np.floor(N * float(top_frac))))
    k_bot = max(1, int(np.floor(N * float(bottom_frac))))

    idx_sort = np.argsort(s)
    short_idx = idx_sort[:k_bot]
    long_idx = idx_sort[-k_top:]

    w = np.zeros(N, float)
    # equal-weight long/short (you can swap to score-weighted)
    w[long_idx] = 1.0 / k_top
    w[short_idx] = -1.0 / k_bot

    # scale to target gross
    gross = np.sum(np.abs(w))
    if gross > 0:
        w = w * (float(gross_leverage) / gross)
    return w


# ===================================================
# Chapter 16 — Transaction Costs, Turnover, Trading
# ===================================================

def ch16_quadratic_transaction_cost(
    trade: np.ndarray,
    linear_cost: float = 0.0,
    quadratic_cost: float = 0.0,
) -> float:
    r"""
    Chapter 16 核心: 交易成本/冲击让“纸面最优”变成“可实现最优”, 并直接影响 TC。

    一个常见最小模型(示意): 
        cost(trade) = c1 * sum(|Δw|) + c2 * sum((Δw)^2)

    - 线性项: 点差/手续费/部分冲击近似
    - 二次项: 冲击随交易量非线性上升

    Example
    -------
    >>> dw=np.array([0.01,-0.02,0.00])
    >>> ch16_quadratic_transaction_cost(dw, linear_cost=10e-4, quadratic_cost=5e-2)
    """
    dw = np.asarray(trade, float).reshape(-1)
    c1 = float(linear_cost)
    c2 = float(quadratic_cost)
    return float(c1 * np.sum(np.abs(dw)) + c2 * np.sum(dw * dw))


# ==================================
# Chapter 17 — Performance Analysis
# ==================================

def ch17_brinson_attribution_two_bucket(
    w_p: np.ndarray,
    w_b: np.ndarray,
    r: np.ndarray,
    group: np.ndarray,
) -> Dict[str, float]:
    r"""
    Chapter 17 核心: 绩效分析/归因: 你赚到的钱, 来自哪里？
    书里更广, 但最常见入口是 Brinson 类归因(配置效应/选股效应/交互项)。

    这里给一个“按 group 分桶”的最简实现(支持任意多个组): 
    - 先计算每组基准权重、组合权重、组收益
    - Allocation: (W_p,g - W_b,g) * R_b,g
    - Selection:  W_p,g * (R_p,g - R_b,g)
    (具体细节不同教材有多种拆法; 这里提供可跑的框架骨架。)

    Example
    -------
    >>> w_p=np.array([0.6,0.4])
    >>> w_b=np.array([0.5,0.5])
    >>> r  =np.array([0.02, -0.01])
    >>> g  =np.array([0, 1])
    >>> ch17_brinson_attribution_two_bucket(w_p,w_b,r,g)
    """
    w_p = np.asarray(w_p, float).reshape(-1)
    w_b = np.asarray(w_b, float).reshape(-1)
    r = np.asarray(r, float).reshape(-1)
    g = np.asarray(group).reshape(-1)

    groups = np.unique(g)
    alloc = 0.0
    sel = 0.0

    for gg in groups:
        idx = (g == gg)
        Wp = float(np.sum(w_p[idx]))
        Wb = float(np.sum(w_b[idx]))
        # group returns: weight-normalized within group
        Rp = float(np.sum(w_p[idx] * r[idx]) / max(Wp, 1e-12))
        Rb = float(np.sum(w_b[idx] * r[idx]) / max(Wb, 1e-12))

        alloc += (Wp - Wb) * Rb
        sel += Wp * (Rp - Rb)

    return {"allocation": float(alloc), "selection": float(sel), "total_active": float(alloc + sel)}


# ==============================
# Chapter 18 — Asset Allocation
# ==============================

def ch18_optimal_allocation_across_strategies(
    expected_active_return: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    r"""
    Chapter 18 核心: 资产配置/策略配置本质上也是 mean-variance, 
    只不过“资产”变成了策略/alpha sleeves/资产类别。

        w* = (1/λ) Σ^{-1} μ

    Example
    -------
    >>> mu=np.array([0.04,0.02])
    >>> S =np.array([[0.09,0.02],[0.02,0.04]])
    >>> ch18_optimal_allocation_across_strategies(mu,S, risk_aversion=2.0)
    """
    mu = _to_col(np.asarray(expected_active_return, float))
    S = _psd_project(covariance)
    lam = float(risk_aversion)
    w = (1.0 / max(lam, 1e-12)) * (_safe_inv(S) @ mu)
    return w.reshape(-1)


# ==============================
# Chapter 19 — Benchmark Timing
# ==============================

def ch19_active_tilt_around_benchmark(
    w_b: np.ndarray,
    active_signal: np.ndarray,
    active_budget: float,
) -> np.ndarray:
    r"""
    Chapter 19 核心: 围绕基准做“可控”的主动偏离(timing/tilt), 把主动风险预算显式化。
    一个最小实现: 把 active_signal 标准化成 sum(|w_a|)=1, 再乘以 active_budget。

    Example
    -------
    >>> wb=np.array([0.5,0.5])
    >>> s =np.array([1.0,-0.5])
    >>> ch19_active_tilt_around_benchmark(wb,s, active_budget=0.1)
    """
    wb = np.asarray(w_b, float).reshape(-1)
    s = np.asarray(active_signal, float).reshape(-1)
    wa = s.copy()
    gross = np.sum(np.abs(wa))
    if gross > 0:
        wa = wa / gross
    wa = wa * float(active_budget)
    # keep active sum ~ 0 by de-meaning (optional)
    wa = wa - wa.mean()
    return wb + wa


# ==============================================
# Chapter 20 — Historical Record for Active Mgmt
# ==============================================

def ch20_skill_vs_luck_t_test(
    active_returns: np.ndarray,
) -> Dict[str, float]:
    r"""
    Chapter 20 核心: 历史记录告诉你“主动很难”, 并且需要区分 skill vs luck。
    最小统计检验: 均值是否显著偏离 0(t-stat)。

        t = mean / (std/sqrt(n))

    Example
    -------
    >>> ar=np.random.randn(1000)*0.01 + 0.0002
    >>> ch20_skill_vs_luck_t_test(ar)["t"]
    """
    a = np.asarray(active_returns, float).reshape(-1)
    n = a.size
    if n < 2:
        return {"mean": float(np.mean(a)) if n else 0.0, "t": 0.0, "n": float(n)}
    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    t = mu / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"mean": mu, "t": float(t), "n": float(n)}


# ==========================
# Chapter 21 — Open Questions
# ==========================

def ch21_research_backlog(
    questions: Tuple[str, ...] = (
        "How stable is IC out-of-sample under regime change?",
        "How do constraints and costs map into transfer coefficient (TC) quantitatively?",
        "How to jointly model alpha decay + market impact to choose optimal horizon?",
        "When do nonlinear forecasts (trees/NN) improve IC after costs?",
        "How to prevent overfitting in high-dimensional cross-sectional signals?",
    ),
) -> Tuple[str, ...]:
    """
    Chapter 21 核心: 这本书是“体系”, 不是“终点”。
    它把大量问题留给你: 信息从哪来、如何衰减、如何实施、如何在真实摩擦下仍然为正。

    这个函数不做计算, 只把“系统性研究议程”显式化, 方便你在工程里变成 backlog。

    Example
    -------
    >>> ch21_research_backlog()[:2]
    """
    return questions


# ====================
# Chapter 22 — Summary
# ====================

@dataclass
class APMSystem:
    """
    Chapter 22(以及全书的“系统观”)的最小可调用封装。

    你可以把它当成一个“主动投资流水线”: 
    1) 用信息生成 alpha(forecast_alpha)
    2) 用风险模型生成 Σ(build_Sigma)
    3) 用优化器生成目标权重(optimize)
    4) 用交易成本与换手规则把目标权重变成可实现权重(trade)
    5) 用绩效评估把结果回灌到 IC/TC/BR 的估计(evaluate)

    下面的 run_one_step 展示了“一次再平衡”的最小闭环。
    """
    forecast_alpha: Callable[[np.ndarray], np.ndarray]
    build_Sigma: Callable[[np.ndarray], np.ndarray]
    optimize: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]  # (alpha, Sigma, w_b)->w*
    transaction_cost: Callable[[np.ndarray], float]
    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def run_one_step(
        self,
        features: np.ndarray,
        w_current: np.ndarray,
        w_benchmark: np.ndarray,
        max_trade: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        运行一次“从信号到权重到交易成本”的闭环。

        Parameters
        ----------
        features : 生成 alpha / risk 的输入(你自己定义: 因子、天气信号、基本面等)
        w_current : 当前持仓
        w_benchmark : 基准持仓
        max_trade : 可选, 把单次换手限制在某个阈值(非常常见的实现摩擦)

        Returns
        -------
        dict: 包含 alpha, Sigma, target_w, traded_w, trade, cost

        Example
        -------
        >>> # 伪例子: 线性 alpha + 恒定 Sigma
        >>> def alpha_fn(X): return ch10_linear_forecast(X, np.array([0.02,-0.01]))
        >>> def sigma_fn(X): return np.eye(X.shape[0])*0.04
        >>> def opt(a,S,wb): return ch14_mean_variance_active_optimal_weights(a,S, risk_aversion=2.0, w_b=wb)
        >>> sys=APMSystem(alpha_fn, sigma_fn, opt,
        ...               lambda dw: ch16_quadratic_transaction_cost(dw, linear_cost=1e-3, quadratic_cost=0.0))
        >>> X=np.eye(3)  # toy
        >>> out=sys.run_one_step(X, w_current=np.zeros(3), w_benchmark=np.zeros(3))
        """
        alpha = self.forecast_alpha(features)
        Sigma = self.build_Sigma(features)
        w_star = self.optimize(alpha, Sigma, w_benchmark)

        if self.postprocess is not None:
            w_star = self.postprocess(w_star)

        trade = w_star - np.asarray(w_current, float)

        if max_trade is not None:
            # simple turnover cap: scale down the trade vector if too large
            turnover = float(np.sum(np.abs(trade)))
            cap = float(max_trade)
            if turnover > cap and turnover > 0:
                trade = trade * (cap / turnover)

        cost = self.transaction_cost(trade)
        w_traded = np.asarray(w_current, float) + trade

        return {
            "alpha": alpha,
            "Sigma": Sigma,
            "target_w": w_star,
            "trade": trade,
            "cost": cost,
            "traded_w": w_traded,
        }


def ch22_build_apm_system_minimal(
    alpha_weights: np.ndarray,
    factor_B: np.ndarray,
    factor_F: np.ndarray,
    idio_D: np.ndarray,
    risk_aversion: float = 1.0,
    linear_cost: float = 0.0,
    quadratic_cost: float = 0.0,
) -> APMSystem:
    """
    Chapter 22 核心: 把全书拼成一个“可运行”的主动投资系统。

    这里给你一个最小装配方案: 
    - Chapter 10: 线性预测 alpha = X w
    - Chapter 3 : 因子风险模型 Σ = B F B' + D
    - Chapter 14: 均值-方差最优权重
    - Chapter 16: 交易成本模型

    注意: 这只是“骨架”。真实系统会加入: 
    约束(行业/国家/因子暴露/杠杆/集中度)、持仓边界、冲击模型、分层执行、风险预算等。

    Example
    -------
    >>> # features (N, P); 这里为了示意, 把 N 当作资产数
    >>> N,P,K=5,2,2
    >>> X=np.random.randn(N,P)
    >>> w_alpha=np.array([0.02,-0.01])
    >>> B=np.random.randn(N,K); F=np.eye(K)*0.02; D=np.eye(N)*0.03
    >>> sys=ch22_build_apm_system_minimal(w_alpha,B,F,D,risk_aversion=2.0, linear_cost=1e-3)
    >>> out=sys.run_one_step(X, w_current=np.zeros(N), w_benchmark=np.zeros(N))
    """
    alpha_weights = np.asarray(alpha_weights, float)

    def alpha_fn(X: np.ndarray) -> np.ndarray:
        return ch10_linear_forecast(X, alpha_weights)

    # In practice, B/F/D may depend on time/features; here kept fixed for clarity.
    def sigma_fn(_: np.ndarray) -> np.ndarray:
        return ch03_factor_risk_model(factor_B, factor_F, idio_D)

    def optimizer(a: np.ndarray, S: np.ndarray, wb: np.ndarray) -> np.ndarray:
        return ch14_mean_variance_active_optimal_weights(
            a, S, risk_aversion=risk_aversion, w_b=wb, enforce_dollar_neutral_active=True
        )

    def tc(dw: np.ndarray) -> float:
        return ch16_quadratic_transaction_cost(dw, linear_cost=linear_cost, quadratic_cost=quadratic_cost)

    return APMSystem(
        forecast_alpha=alpha_fn,
        build_Sigma=sigma_fn,
        optimize=optimizer,
        transaction_cost=tc,
        postprocess=None,
    )