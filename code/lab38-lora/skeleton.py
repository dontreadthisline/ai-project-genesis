"""Lab 38 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class LoRALinear:
    def __init__(self, base_linear, r, alpha): ...   # 冻结 base
    def forward(self, x): ...                        # base(x) + scale * B(A(x))
def apply_lora(model, targets, r, alpha): ...        # 替换 q/k/v/o 投影
def merge_lora(model): ...                           # W += BA 合并（推理零开销）

