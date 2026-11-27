"""Lab 14 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def attention(q, k, v, mask=None): ...     # 1.1 公式，softmax 数值稳定
class MultiHeadAttention: ...
class Block: ...                            # 1.4 pre-norm
class MiniTransformer: ...                  # emb + pos + L*Block + ln_f + head

