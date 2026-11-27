"""Lab 36 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 采样等价性：cached 与 naive 生成在 greedy（T=0）下逐 token 一致
# # T2: 温度单调性：同一分布下采样熵随 T 单调增（MC 估计）
# # T3: top-p 自适应性：尖分布（熵 0.5）保留的候选数 < 平分布（熵 3）的
# # T4: 缓存正确性：手工前向一步的 K/V 与 cache 追加后一致
# # T5: 数值一致：T=1、top_k=V 时采样分布 == 原分布（频率检验）





# T1: 采样等价性：cached 与 naive 生成在 greedy（T=0）下逐 token 一致
# # T2: 温度单调性：同一分布下采样熵随 T 单调增（MC 估计）
# # T3: top-p 自适应性：尖分布（熵 0.5）保留的候选数 < 平分布（熵 3）的
# # T4: 缓存正确性：手工前向一步的 K/V 与 cache 追加后一致
# # T5: 数值一致：T=1、top_k=V 时采样分布 == 原分布（频率检验）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
