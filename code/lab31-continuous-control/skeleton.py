"""Lab 31 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class DeterministicActor: ...   # -> tanh -> [-2, 2]
class DoubleCritic: ...         # Q1, Q2
def ddpg_td3_step(batch, nets, cfg): ...    # cfg 切换三处修复
class SACActor: ...             # 高斯 + tanh 压缩（log_prob 的 tanh 修正项）
def sac_step(batch, nets, alpha, cfg): ...

