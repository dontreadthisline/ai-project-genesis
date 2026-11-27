"""Lab 43 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def record_trajectory(env, model, task, cfg) -> dict: ...
def batch_collect(tasks, n_rollouts, cfg) -> list[dict]: ...
def traj_stats(trajs) -> dict: ...          # 1.2 全部指标
def dump_jsonl(trajs, path): ...            # JSONL 落盘
def classify_failure(traj) -> str: ...      # 1.3 分类树

