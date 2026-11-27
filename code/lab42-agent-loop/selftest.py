"""Lab 42 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 工具边界：calc("__import__('os')") 返回错误串而非执行；py 超时返回 TIMEOUT
# T2: 环境全响应：对 20 种任意乱码 action，env.step 均返回定义好的观测、不抛异常
# T3: 终止语义：answer 后 done=True；max_steps 截断 done=True 且 info 标记 truncated
# T4: 上下文一致性：同一交互序列两次拼接的 token 序列完全一致（确定性协议）
# T5: 判分器：构造对/错/格式错答案，score ∈ {1, 0} 边界清晰





# T1: 工具边界：calc("__import__('os')") 返回错误串而非执行；py 超时返回 TIMEOUT
# T2: 环境全响应：对 20 种任意乱码 action，env.step 均返回定义好的观测、不抛异常
# T3: 终止语义：answer 后 done=True；max_steps 截断 done=True 且 info 标记 truncated
# T4: 上下文一致性：同一交互序列两次拼接的 token 序列完全一致（确定性协议）
# T5: 判分器：构造对/错/格式错答案，score ∈ {1, 0} 边界清晰



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
