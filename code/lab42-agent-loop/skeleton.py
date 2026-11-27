"""Lab 42 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class ToolEnv:
    def __init__(self, tasks): ...
    def reset(self) -> str: ...               # 返回任务 prompt
    def step(self, action_str) -> (obs, done, info): ...
    def score(self, final_answer) -> float: ...  # 可验证判分
def agent_loop(model, env, max_steps, sampler): ...
    # 拼接 -> 解码动作段 -> env.step -> 拼 observation -> 循环

