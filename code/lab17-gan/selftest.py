"""Lab 17 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: D* 公式数值验证：两个解析高斯 p_data=N(0,1), p_g=N(2,1)，
#     在网格点上 D*(x) = p_data/(p_data+p_g) 与公式一致（这是固定 G 的极限 sanity）
# T2: BCE with logits 数值稳定：大 logits 不产生 nan
# T3: 训练早期的均衡基准：D 输出接近 0.5 时 D 的二分类准确率应接近 50%~70% 区间
# T4: 生成器输出范围 [0,1]（sigmoid 生效）





# T1: D* 公式数值验证：两个解析高斯 p_data=N(0,1), p_g=N(2,1)，
#     在网格点上 D*(x) = p_data/(p_data+p_g) 与公式一致（这是固定 G 的极限 sanity）
# T2: BCE with logits 数值稳定：大 logits 不产生 nan
# T3: 训练早期的均衡基准：D 输出接近 0.5 时 D 的二分类准确率应接近 50%~70% 区间
# T4: 生成器输出范围 [0,1]（sigmoid 生效）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
