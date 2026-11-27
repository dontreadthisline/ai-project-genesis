"""Lab 08 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def full_batch_grad(model, data): ...            # 全数据集梯度（参考真值）
def batch_grad(model, data, idx): ...
def grad_stats(model, data, sizes, n_rep):  # 返回各 batch size 的方差、SNR、cos
    ...

