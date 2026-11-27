"""Lab 32 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 往返恒等：decode(encode(x)) == x（100 段随机文本子串，含中文/数字/标点）
# # T2: 合并确定性：同语料两次训练结果一致；打乱合并顺序编码则不同
# T3: 词表单调性：vocab_size 增大时 token 数单调不增
# T4: 频率正确性：第一次合并的 token 对是语料中最高频相邻对（手工小语料对照）





# T1: 往返恒等：decode(encode(x)) == x（100 段随机文本子串，含中文/数字/标点）
# # T2: 合并确定性：同语料两次训练结果一致；打乱合并顺序编码则不同
# T3: 词表单调性：vocab_size 增大时 token 数单调不增
# T4: 频率正确性：第一次合并的 token 对是语料中最高频相邻对（手工小语料对照）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
