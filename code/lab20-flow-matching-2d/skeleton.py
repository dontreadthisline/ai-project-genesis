"""Lab 20 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class VelocityNet: ...                 # (t, x) -> v，t 用 sinusoidal 嵌入
def cfm_loss(model, x1): ...      # 1.3 直线路径损失
def euler_sample(model, n_steps): ...

