"""Lab 14 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 因果掩码：注意力权重矩阵下三角严格为 0（p < 1e-8），每行和为 1
# T2: 置换等变性验证（无位置编码、无掩码时）：打乱序列顺序，输出对应重排
# T3: 多头形状：Concat 后维度 == d_model
# T4: 参数量手算：4/d² 系数核对（每 Block ≈ 12d²：4 个 d² 投影 + FFN 8d²）





# T1: 因果掩码：注意力权重矩阵下三角严格为 0（p < 1e-8），每行和为 1
# T2: 置换等变性验证（无位置编码、无掩码时）：打乱序列顺序，输出对应重排
# T3: 多头形状：Concat 后维度 == d_model
# T4: 参数量手算：4/d² 系数核对（每 Block ≈ 12d²：4 个 d² 投影 + FFN 8d²）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
