"""Lab 37 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

def build_sft_dataset(tasks, n): ...       # (prompt_ids, completion_ids)
def sft_collate(batch, pad_id): ...        # labels: prompt 部分全 -100
def sft_train(model, loader, cfg): ...     # 与 pretrain 共用循环，换 loader

