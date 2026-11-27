"""Lab 16 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def sf_grad(f, mu, sigma, n): ...
    # (f(z) - b) * (z-mu)/sigma^2 对 mu 的均值；b 为可选 baseline（默认 0）
def pathwise_grad(f, mu, sigma, n): ...
    # eps ~ N(0,1), z = mu + sigma*eps, 数值梯度 df/dz (中心差分) * 1

