"""Lab 34 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 参数量手算对照：按 1.2 公式 vs 实现实际参数数（误差 < 1%）
# # T2: 因果性：位置 t 的 logits 不受未来 token 影响（修改 x[t+1:] 后 logits[:, :t+1] 不变）
# # T3: 形状链：emb (B,T,d) -> blocks -> logits (B,T,V) 逐层形状检查
# # T4: 随机初始化生成 = 均匀采样分布（每 token 的边缘分布接近均匀）
# # T5: 单步可训：随机小批上 loss 能从 ln(V) 下降（几步 SGD）





# T1: 参数量手算对照：按 1.2 公式 vs 实现实际参数数（误差 < 1%）
# # T2: 因果性：位置 t 的 logits 不受未来 token 影响（修改 x[t+1:] 后 logits[:, :t+1] 不变）
# # T3: 形状链：emb (B,T,d) -> blocks -> logits (B,T,V) 逐层形状检查
# # T4: 随机初始化生成 = 均匀采样分布（每 token 的边缘分布接近均匀）
# # T5: 单步可训：随机小批上 loss 能从 ln(V) 下降（几步 SGD）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
