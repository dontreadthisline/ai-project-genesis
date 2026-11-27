"""Lab 45 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def collect_group_rollouts(model, env, prompts, G, cfg): ...
    # 1.prompt 采样 2.agent rollout 3.轨迹收集（含逐步 logprob）
def compute_group_reward(trajs, reward_cfg): ...     # 4.奖励计算
def compute_advantages(rewards): ...                 # 5.组归一化
def grpo_update(model, ref_model, trajs, advs, cfg): ...  # 6.token 级更新
def evaluate(model, env, tasks, n): ...              # 7.独立评估（真实成功率）
def train_loop(cfg): ...   # 以上循环 + KL 监控 + hack 监控 + checkpoint

