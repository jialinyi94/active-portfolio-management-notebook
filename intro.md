# Grinold & Kahn (Active Portfolio Management) — “one function per chapter”

I am a book about Active Portfolio Management! Wikipedia has [information about active management](https://en.wikipedia.org/wiki/Active_management).

% An admonition containing a note
:::{note}
This is a runnable note for Active Portfolio Management
:::

这份文件的目标不是“把整章抄成代码”，而是把每章最核心的概念，压缩成一个可调用的
Python 函数：输入是什么、输出是什么、背后在做什么、以及一个最小示例。

你可以把它当成：
- 一个“主动投资系统”的可执行接口草图（alpha -> risk -> optimize -> trade -> evaluate）
- 一套统一的符号语言（便于你把书里的逻辑接到你自己的研究/交易框架上）

## 整本书推崇的“主动投资系统”，应该如何调用这些函数？

把它当成一个流水线（你在商品/电力/股票都能套用）：

1. 定义“共识/基准”（Chapter 2/4）

    - 你要相对谁做主动？CAPM/APT 给的是“共识回报”，Chapter 4 定义了你最终记分的 active return（相对基准）。

2. 搭风险底盘 $\Sigma$（Chapter 3）

    - 用 `ch03_factor_risk_model(B,F,D)`（或者你自己的风险模型）得到协方差矩阵。
    - 这是后面所有“风险预算/约束/可实现性”的共同语言。

3.	把信息变成 alpha，并测 IC/衰减（Chapter 10–13）

    - `ch10_linear_forecast(X,w)` 先做最小原型；
    - 用 `ch11_information_coefficient(forecast, realized)` 回测评估信号质量；
    - 用 `ch13_ic_decay(ic0, half_life, dt)` 把信息衰减编码进你的持有期/换手假设；
    - 用 `ch12_ic_to_expected_ir(ic, breadth, tc)` 把统计量翻译成“你大概能做到多强”。

4.	优化构建组合（Chapter 14/15/18/19）

    - 最小：`ch14_mean_variance_active_optimal_weights(alpha, Sigma, λ, w_b)` 得到目标权重；
    - 若你做 long/short：`ch15_long_short_from_scores(scores, gross, top, bottom)`；
    - 多策略/多资产：`ch18_optimal_allocation_across_strategies(mu, cov, λ)`；
    - 围绕基准做小偏离：`ch19_active_tilt_around_benchmark(wb, signal, budget)`。

5.	把“纸面最优”变成“可交易最优”（Chapter 16）

    - 交易成本用 `ch16_quadratic_transaction_cost(Δw, c1, c2)`，并通过换手/仓位约束影响 TC。

6.	绩效归因与统计检验（Chapter 17/20）

    - 用 `ch17_brinson_attribution_two_bucket(...)` 看收益来自配置还是选股（或你定义的分组）；
    - 用 `ch20_skill_vs_luck_t_test(active_returns)` 判断你看到的 IR/alpha 是否可能只是噪声。

## Function signatures

```{literalinclude} src/apm/core/utils.py
:lineno-match:
```