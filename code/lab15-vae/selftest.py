"""Lab 15 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: KL 闭式对照：随机制 (mu, logvar) 手算 1.3 公式 == vae_loss 的 kl 项（1e-6）
# T2: logvar 数值稳定：encoder 输出 clamp 到 [-10, 10]，exp 不上溢
# T3: ELBO ≤ log p 的方向检查：训练后 ELBO 单调上升
# T4: 重参数化梯度存在：对 mu 的数值梯度 == 解析梯度（单样本噪声固定时，rel err < 1e-5）





# T1: KL 闭式对照：随机制 (mu, logvar) 手算 1.3 公式 == vae_loss 的 kl 项（1e-6）
# T2: logvar 数值稳定：encoder 输出 clamp 到 [-10, 10]，exp 不上溢
# T3: ELBO ≤ log p 的方向检查：训练后 ELBO 单调上升
# T4: 重参数化梯度存在：对 mu 的数值梯度 == 解析梯度（单样本噪声固定时，rel err < 1e-5）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
