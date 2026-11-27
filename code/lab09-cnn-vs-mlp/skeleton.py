"""Lab 09 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class Conv2d:   # weight (C_out, C_in, K, K)
    def forward(self, x): ...    # x (N, C_in, H, W)
    def backward(self, grad): ...
class Pool2d:   # 非重叠 max/avg，无参数
class Flatten: ...

