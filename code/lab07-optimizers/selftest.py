"""Lab 07 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 常数梯度下 Adam 第 1 步更新量 ≈ lr（偏差修正生效）：
#     g = 0.1 恒定，step 后 |Δθ| ≈ lr * 1.0 ± 1e-4
# T2: Momentum 对交替梯度 (+g, -g)：θ 几乎不动（v≈0）；对恒定梯度：步长按 1/(1-β) 放大
# T3: AdamW 与 Adam 在 weight_decay>0 时轨迹不同；decay=0 时一致
# T4: cosine_schedule 端点：t=0 为 0（warmup 起点）、t=T 为 0、峰值在 t=T_w





# T1: 常数梯度下 Adam 第 1 步更新量 ≈ lr（偏差修正生效）：
#     g = 0.1 恒定，step 后 |Δθ| ≈ lr * 1.0 ± 1e-4
# T2: Momentum 对交替梯度 (+g, -g)：θ 几乎不动（v≈0）；对恒定梯度：步长按 1/(1-β) 放大
# T3: AdamW 与 Adam 在 weight_decay>0 时轨迹不同；decay=0 时一致
# T4: cosine_schedule 端点：t=0 为 0（warmup 起点）、t=T 为 0、峰值在 t=T_w



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
