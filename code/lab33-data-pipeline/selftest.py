"""Lab 33 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: shift 正确性：target[i] == input[i+1]（逐位）
# T2: pad 不计损失：全 pad 批的 loss == 0（0/0 以 0 处理）
# T3: 掩码独立性：把 target 的后半替换为随机 token，前半位置的 loss 不变
#     （位置 < k 的预测不依赖 target > k —— 验证无泄漏）
# T4: packing 计数：N 个文档 pack 后总 token 利用率 > 95%
# T5: 文档边界：pack 后第一个文档的 attention 最后一行不含第二文档 token





# T1: shift 正确性：target[i] == input[i+1]（逐位）
# T2: pad 不计损失：全 pad 批的 loss == 0（0/0 以 0 处理）
# T3: 掩码独立性：把 target 的后半替换为随机 token，前半位置的 loss 不变
#     （位置 < k 的预测不依赖 target > k —— 验证无泄漏）
# T4: packing 计数：N 个文档 pack 后总 token 利用率 > 95%
# T5: 文档边界：pack 后第一个文档的 attention 最后一行不含第二文档 token



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
