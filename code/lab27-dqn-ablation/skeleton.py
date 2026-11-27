"""Lab 27 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def ddqn_target(q_online, q_target, s_next, r, gamma, done): ...   # 1.2
class DuelingQNet: ...   # 共享 trunk + V head + A head

