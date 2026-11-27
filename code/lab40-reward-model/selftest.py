"""Lab 40 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: loss 平衡点：r_w == r_l 时 loss == ln2；r_w - r_l = 5 时 loss < 0.01
# T2: 标量头形状：输出 (B,) 而非 (B, T)；只取末位（completion 结束 token）
# T3: 梯度加权：构造 (r_w - r_l) ∈ {0, 5} 两类，前者的梯度范数远大
#     （logistic 的「难样本加权」特性，T3 数值验证 > 10 倍）
# T4: held-out 判别：训练后 AUC > 0.85（构造数据）





# T1: loss 平衡点：r_w == r_l 时 loss == ln2；r_w - r_l = 5 时 loss < 0.01
# T2: 标量头形状：输出 (B,) 而非 (B, T)；只取末位（completion 结束 token）
# T3: 梯度加权：构造 (r_w - r_l) ∈ {0, 5} 两类，前者的梯度范数远大
#     （logistic 的「难样本加权」特性，T3 数值验证 > 10 倍）
# T4: held-out 判别：训练后 AUC > 0.85（构造数据）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
