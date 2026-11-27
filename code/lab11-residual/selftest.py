"""Lab 11 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: ResidualBlock 在 F=0（零初始化最后 conv）时输出 == 输入
# T2: 随机权重下，反向传播到第 1 层的梯度范数：res 网络随深度下降缓慢、
#     plain 网络显著更小（深度 20 对照，比值 > 5 倍）
# T3: 深度增加时前向激活范数：res 网络各层范数稳定，plain 网络可能衰减





# T1: ResidualBlock 在 F=0（零初始化最后 conv）时输出 == 输入
# T2: 随机权重下，反向传播到第 1 层的梯度范数：res 网络随深度下降缓慢、
#     plain 网络显著更小（深度 20 对照，比值 > 5 倍）
# T3: 深度增加时前向激活范数：res 网络各层范数稳定，plain 网络可能衰减



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
