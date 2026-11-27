"""Lab 43 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 自包含重放：从轨迹重放动作序列，重建的观测序列逐项一致
# T2: 指标自洽：success_rate + failure 分类占比和 == 1
# T3: 循环检测：构造重复动作轨迹被正确标记
# T4: 成功率区间：n=200 时标准误 ≈ sqrt(0.25/200) ≈ 0.035，
#     报告值应带 ±2se 区间





# T1: 自包含重放：从轨迹重放动作序列，重建的观测序列逐项一致
# T2: 指标自洽：success_rate + failure 分类占比和 == 1
# T3: 循环检测：构造重复动作轨迹被正确标记
# T4: 成功率区间：n=200 时标准误 ≈ sqrt(0.25/200) ≈ 0.035，
#     报告值应带 ±2se 区间



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
