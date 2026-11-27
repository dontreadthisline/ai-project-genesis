"""Lab 37 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 掩码正确性：labels 中 prompt 段全为 -100，completion 段正确 shift
# T2: 损失归一化：分母 = completion token 数（不是序列总长）
# T3: 泛化检查：模板 A 训练、模板 B 测试的准确率差（模板过拟合信号）
# T4: 基线对照：未 SFT 的基座模型对指令的回答形态（续写而非作答）





# T1: 掩码正确性：labels 中 prompt 段全为 -100，completion 段正确 shift
# T2: 损失归一化：分母 = completion token 数（不是序列总长）
# T3: 泛化检查：模板 A 训练、模板 B 测试的准确率差（模板过拟合信号）
# T4: 基线对照：未 SFT 的基座模型对指令的回答形态（续写而非作答）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
