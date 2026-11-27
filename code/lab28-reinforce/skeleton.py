"""Lab 28 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class PolicyNet: ...                       # 4 -> 128 -> 128 -> 4 logits
def reinforce(env_fn, episodes, lr, gamma, use_baseline): ...
def policy_grad_loss(logits, actions, returns_or_adv): ...   # -logπ * A 的均值

