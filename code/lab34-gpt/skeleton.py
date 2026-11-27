"""Lab 34 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class GPT:
    def __init__(self, vocab, d, n_head, n_layer, block): ...
    def forward(self, idx): ...        # (B, T) -> logits (B, T, V)
    def loss(self, idx, targets): ...  # 交叉熵均值
    def generate(self, idx, max_new, temperature, top_k): ...
    # MLX 默认不构建梯度图（仅 mx.grad 会），无需 no_grad 等价物；
# 生成循环内对 logits 先 mx.eval() 再采样，避免惰性堆积

