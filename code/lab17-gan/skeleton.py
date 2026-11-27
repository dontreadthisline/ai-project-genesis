"""Lab 17 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class Generator:     # z(64) -> 256 -> 512 -> 784 sigmoid
class Discriminator: # 784 -> 512 -> 256 -> 1 sigmoid
def d_loss(real_logits, fake_logits): ...   # BCE with logits
def g_loss(fake_logits): ...

