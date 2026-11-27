"""Lab 12 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: BN train 模式：输出各列均值≈0、方差≈1；running stats 随 batch 更新
#     （手算一步：0.9*0 + 0.1*mu_batch）
# T2: BN eval 模式用 running stats：同一输入两次前向输出一致（train 模式则不同）
# T3: LN：每行均值≈0 方差≈1（与 batch 无关，batch=1 也成立）
# T4: dropout 期望保持：大样本下 mean(drop(x)) ≈ mean(x)（相对误差 < 2%）





# T1: BN train 模式：输出各列均值≈0、方差≈1；running stats 随 batch 更新
#     （手算一步：0.9*0 + 0.1*mu_batch）
# T2: BN eval 模式用 running stats：同一输入两次前向输出一致（train 模式则不同）
# T3: LN：每行均值≈0 方差≈1（与 batch 无关，batch=1 也成立）
# T4: dropout 期望保持：大样本下 mean(drop(x)) ≈ mean(x)（相对误差 < 2%）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
