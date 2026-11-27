"""Lab 15 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class Encoder: ...    # 输出 mu (B,2), logvar (B,2)
class Decoder: ...
def reparameterize(mu, logvar): ...
def vae_loss(x, recon, mu, logvar, beta=1.0): ...   # 返回 (loss, recon_term, kl_term)

