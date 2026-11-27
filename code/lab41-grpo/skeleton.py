"""Lab 41 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def make_arith_task(): ...             # 题目 + 可验证判分器
def group_rollout(model, prompts, G, cfg): ...   # 批采样 G 个回答 + logprob_old
def grpo_advantage(rewards): ...          # 1.1 归一化
def grpo_loss(...): ...                   # token 级 clip 目标 + KL(ref)

