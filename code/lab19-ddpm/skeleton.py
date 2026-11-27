"""Lab 19 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class Diffusion:
    def betas_schedule(T): ...          # 线性或余弦
    def q_sample(self, x0, t, eps): ... # 1.1 边缘采样
    def loss(self, x0): ...        # 1.2 simple loss
    def sample(self, n): ...       # 1.3 完整链
class EpsNet: ...                        # 简化 UNet：conv + 时间嵌入（sinusoidal）

