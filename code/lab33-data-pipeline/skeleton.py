"""Lab 33 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def make_shift_pairs(ids, block_size): ...    # 滑窗切 -> (input, target)
def collate(batch, pad_id): ...               # pad + labels(-100)
def pack_sequences(docs, block_size): ...     # 拼接 + doc_mask
def masked_ce_loss(logits, labels): ...       # 1.3 公式

