"""Lab 21 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class TimeEmbedding: ...               # sinusoidal -> MLP
class ResBlockT: ...                   # ResBlock + 时间 FiLM
class UNetLite: ...                    # 下采样-上采样 + 跳跃，输入 (x, t)
def cfm_loss(model, x1): ...      # 同 Lab 20，域换成图像
def sample(model, n, steps): ...  # 欧拉积分 + 最后一步 clamp 到 [-1,1]

