"""Lab 04 度量模块：Entropy / Cross Entropy / KL / Brier（MLX 实现）。

实现要求见 docs/stage1-probability/lab04-entropy-kl.md（§1.1、§1.4、§2）。
全部使用自然对数（nats）。输入为概率向量（mx.array，建议 fp64）。
"""
import mlx.core as mx


def entropy(p) -> float:
    """H(P) = -sum P log P（nats）。约定 0*log0 = 0（p=0 的项贡献为 0）。

    0*log0 处理：MX 不支持布尔索引，用 mx.where(mask, p*log_p, 0.0) 后求和。
    """
    raise NotImplementedError("TODO(lab04 §1.1)")


def cross_entropy(p, q) -> float:
    """H(P, Q) = -sum P log Q。

    当 P(x)>0 而 Q(x)=0 时应返回 inf（用 mx.isinf 可测；讲义 §2 对 eps 处理的
    讨论写进报告）。
    """
    raise NotImplementedError("TODO(lab04 §1.1)")


def kl_divergence(p, q) -> float:
    """D_KL(P||Q) = sum P (log P - log Q)。实现方式限定为该形式，p=0 的项跳过。"""
    raise NotImplementedError("TODO(lab04 §1.2)")


def brier_score(q, y: int) -> float:
    """多分类 Brier：sum_k (q_k - 1[k==y])^2。"""
    raise NotImplementedError("TODO(lab04 §1.4)")
