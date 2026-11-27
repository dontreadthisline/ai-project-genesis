"""Lab 10 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def make_permutation( n=784): ...        # 固定置换，训练/测试共用
def permute_dataset(X, perm): ...            # X (N, 28, 28) -> 打乱后 reshape 回

