"""Lab 23 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def build_P_R(env):
    """从 env.unwrapped.P 构造 P (S,A,S), R (S,A)，终止态自吸收处理"""
def expected_reward(R, s, a): ...
def sample_step(P, s, a): ...    # 按 P 采样下一状态（自实现，不用 env.step）

