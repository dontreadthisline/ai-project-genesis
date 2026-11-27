"""Lab 30 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 更新前 r == 1 且 clip 不变：初始时刻目标 == A 的均值
# T2: clip 语义数值：r=2, A>0 时目标锁定 (1+ε)A；r=0.2, A<0 时锁定 (1-ε)A
# T3: GAE 端点：λ=0 时 == δ；λ=1 且 V=0 时 == G_t（手算小例）
# T4: advantage 标准化后均值 0、方差 1（批内）
# T5: KL 早停：构造大策略变化，KL 超阈值时更新被跳过





# T1: 更新前 r == 1 且 clip 不变：初始时刻目标 == A 的均值
# T2: clip 语义数值：r=2, A>0 时目标锁定 (1+ε)A；r=0.2, A<0 时锁定 (1-ε)A
# T3: GAE 端点：λ=0 时 == δ；λ=1 且 V=0 时 == G_t（手算小例）
# T4: advantage 标准化后均值 0、方差 1（批内）
# T5: KL 早停：构造大策略变化，KL 超阈值时更新被跳过



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
