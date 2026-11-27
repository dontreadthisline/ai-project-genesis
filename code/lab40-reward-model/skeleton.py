"""Lab 40 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""

class RewardModel: ...          # GPT 骨干 + scalar head，最后一个 token 位置的输出
def rm_loss(r_w, r_l): ...      # 1.1 公式
def rm_train(model, pairs, cfg): ...
def rm_auc(model, pairs): ...   # 判别 AUC

