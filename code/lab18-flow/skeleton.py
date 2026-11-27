"""Lab 18 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class AffineCoupling:        # 实现 forward / inverse / log_det
class Flow:                  # 堆叠 coupling，交替 mask 两半
    def sample(self, n): ...
    def log_prob(self, x): ...   # 1.1 公式累加 -log|det|

