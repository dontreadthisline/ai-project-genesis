"""Lab 29 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class ActorCritic: ...     # 共享 trunk：策略头 logits + 价值头标量
def a2c_train(env_fn, steps, cfg): ...   # on-policy：采集 n_step 回合批，一次更新

