"""Lab 26 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class QNet: ...                        # 4 -> 128 -> 128 -> 4
class ReplayBuffer: ...                # 环形缓冲：push / sample(batch)
def dqn_train(env_fn, steps, cfg): ... # ε 衰减、目标网络同步、梯度步
def eval_policy(qnet, env, episodes): ...

