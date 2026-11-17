"""Lab 01 绘图辅助。图表规格见讲义 §4（预期图表）。

桥接约定：matplotlib 直接接受 mx.array（mlx 实现了 `__array__` 协议），无需任何转换；
标量取值用 .item()，转 Python 列表用 .tolist()。
"""

def plot_pdf_and_hist(ax, samples, pdf_fn, bins: int = 40, title: str = "") -> None:
    """图 2/图 3 用：样本直方图（density=True）+ 真实 PDF 曲线叠加。

    samples 为 mx.array，可直接传给 ax.hist（零桥接）。
    PDF 曲线的 x 网格取 samples.min()..max() 外扩 10%（mx.linspace 生成）。
    """
    raise NotImplementedError("TODO(lab01 图 2/3): ax.hist(..., density=True) + ax.plot")


def plot_density_ellipses(ax, mu, Sigma, levels=(1.0, 2.0), **style) -> None:
    """图 4/图 5 用：由特征分解画等密度椭圆。

    Sigma = U diag(lambda) U^T，水平 c 的椭圆半轴为 c*sqrt(lambda_i)，方向为特征向量
    （讲义 §1.3）。参数化：theta in [0, 2*pi]，点 = mu + c*(sqrt(l1)*u1*cos + sqrt(l2)*u2*sin)。
    特征分解用 mx.linalg.eigh（Sigma 对称，stream=mx.cpu），画图前桥接。
    """
    raise NotImplementedError("TODO(lab01 图 4/5): 特征分解参数化椭圆")
