"""Lab 30 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class ActorCritic: ...                    # Lab 29 复用
def gae(rewards, values, dones, gamma, lam): ...
def ppo_loss(logits_new, logits_old, actions, adv, returns, cfg): ...
def ppo_train(env_fn, iters, cfg): ...    # 采集 batch -> PPO epoch 多轮更新

