"""Lab 11 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class PlainConvBlock:   # conv-bn-relu-conv-bn + (可选下采样)
class ResidualBlock:    # 同上 + 跳跃连接；下采样时 1x1 conv 投影对齐维度
def make_plain_cnn(depth): ...
def make_res_cnn(depth): ...   # 参数量与 plain 对齐（±15%）

