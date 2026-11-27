# stage5 图定义（被 gen-figs.py exec 加载）

fig("lab32-tokenizer", "bpe-merge", "1.2", 300, "BPE：从字符到子词",
    "每步合并最高频相邻对：频率驱动的贪心压缩。合并顺序即优先级——编码必须按训练顺序应用同样的合并。",
    "低, 最低, 低频 → 合并成块", """
const F = Fig(document.getElementById('fig'));
const steps = [
  ['低', '最', '新', '低', '频', '低', '最'],
  ['最低', '新', '低', '频', '最低'],
  ['最低', '新', '低频', '最低'],
  ['最新', '低频', '最低'],
];
for (let r = 0; r < 4; r++) {
  F.text(80, 60 + r * 62, '步骤 ' + (r + 1) + ':', { size: 15, color: F.P.dim, anchor: 'end' });
  let x = 110;
  for (const tok of steps[r]) {
    const w = 22 + tok.length * 30;
    F.box({ x, y: 38 + r * 62, w, h: 42, label: tok, fill: 'rgba(58,74,168,0.08)', size: 15 });
    x += w + 8;
  }
}
F.text(360, 300, '「最低」高频 → 先合并；token 数单调下降', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab32-tokenizer", "compression-curve", "1.3", 300, "词表大小的边际收益",
    "压缩率随 $|V|$ 快速上升后饱和：高频对耗尽后，继续合并的是低频对——embedding 参数（$|V| \\times d$）线性增长、收益骤减。",
    "压缩率 vs |V|", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 100, x1: 4000, y0: 1, y1: 5, labels: ['词表 |V|', '压缩率（字符/token）'] });
F.curve(v => 4.5 - 3.5 * Math.exp(-0.0011 * (v - 100)), { t0: 100, t1: 4000, color: F.P.ink, width: 2.3, label: 'BPE', labelAt: 2000 });
F.curve(v => 2.4, { t0: 100, t1: 4000, color: F.P.dim, width: 1.3, rough: false, label: '字符级 = 1', labelAt: 2200 });
F.vbar(900, 4.9, { w: 30, fill: 'rgba(0,0,0,0.15)', color: 'rgba(0,0,0,0)', width: 0 });
F.text(900, 1.35, '饱和点', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab33-data-pipeline", "shift-mask", "1.1", 310, "自回归的输入-目标对",
    "target = input 左移一位：位置 $t$ 的预测头只用 $\\le t$ 的输入（因果掩码），一条序列同时是 $T$ 个训练样本；pad 位置 label = $-100$ 不计损。",
    "input/target 错位与掩码", """
const F = Fig(document.getElementById('fig'));
const toks = ['今', '天', '天', '气', '很', '好'];
for (let i = 0; i < 6; i++) {
  F.box({ x: 60 + i * 100, y: 60, w: 88, h: 44, label: toks[i], fill: 'rgba(58,74,168,0.08)' });
  if (i < 5) F.box({ x: 110 + i * 100, y: 150, w: 88, h: 44, label: toks[i + 1], fill: 'rgba(26,127,55,0.08)' });
  else F.box({ x: 610, y: 150, w: 88, h: 44, label: 'pad', fill: 'rgba(0,0,0,0.05)', color: F.P.dim });
}
F.text(40, 82, '输入', { size: 17, color: F.P.ink, anchor: 'end' });
F.text(40, 172, '目标', { size: 17, color: F.P.green, anchor: 'end' });
for (let i = 0; i < 5; i++) F.arrow(104 + i * 100, 104, 154 + i * 100, 150, { width: 1.1, color: F.P.dim });
F.text(360, 240, '位置 t 的目标 = 位置 t+1 的输入（左移一位）', { size: 16, color: F.P.dim, anchor: 'middle' });
F.text(654, 215, 'label = −100', { size: 13, color: F.P.dim, anchor: 'middle' });
""")

fig("lab33-data-pipeline", "packing-boundary", "1.4", 300, "Sequence Packing 的边界",
    "短样本拼进 block 提高利用率（pad 浪费 → 近 0），但文档边界必须加掩码——否则上个文档的结尾在「预测」下个文档的开头。",
    "同 block 两个文档 + 边界掩码", """
const F = Fig(document.getElementById('fig'));
for (let i = 0; i < 12; i++) {
  const doc = i < 7;
  F.box({ x: 70 + i * 50, y: 100, w: 44, h: 44, label: doc ? 'A' + (i + 1) : 'B' + (i - 6),
         fill: doc ? 'rgba(58,74,168,0.1)' : 'rgba(26,127,55,0.1)', size: 13 });
}
F.arrow(70 + 7 * 50 - 4, 160, 70 + 7 * 50 - 4, 96, { color: 'rgba(200,60,60,0.8)', width: 2, label: '边界：不可跨看', labelDy: 24 });
F.text(360, 200, '利用率 > 95%（对照不打包的 ~65%）', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab34-gpt", "gpt-stack", "1.1", 320, "GPT 的组件栈",
    "Embedding + 位置 → $L$ 个 pre-norm Block（因果 MHA + FFN）→ 最终 LN → 权重共享 LM Head——$12Ld^2$ 参数预算的分层账。",
    "自回归 Transformer 的栈", """
const F = Fig(document.getElementById('fig'));
const stack = ['token ids', 'Embedding + 位置', 'Block × L（MHA + FFN, pre-norm）', '最终 LayerNorm', 'LM Head（与 Embedding 共享）'];
for (let i = 0; i < stack.length; i++) {
  F.box({ x: 150, y: 250 - i * 56, w: 380, h: 44, label: stack[i], fill: 'rgba(58,74,168,0.06)', size: 15 });
  if (i < stack.length - 1) F.arrow(340, 250 - i * 56, 340, 250 - i * 56 - 12, { width: 1.4 });
}
F.text(600, 130, 'logits (B,T,V)', { size: 15, color: F.P.deep, anchor: 'start' });
""")

fig("lab34-gpt", "param-budget", "1.2", 310, "参数量：12Ld² 的分层账",
    "每 Block：MHA 四投影 $4d^2$ + FFN $8d^2$ = $12d^2$；FFN 占 2/3；词表项 $|V|d$ 在共享时只计一次——$L=4, d=256$ 时 $\\approx 4.2$M。",
    "参数分解", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 3.2, y0: 0, y1: 9, labels: ['', 'd² 的倍数'] });
F.vbar(0.6, 4, { w: 0.55, fill: 'rgba(58,74,168,0.25)' });
F.vbar(1.6, 8, { w: 0.55, fill: 'rgba(26,127,55,0.25)' });
F.text(0.6, 4.6, 'MHA 4d²', { size: 16, color: F.P.ink, anchor: 'middle' });
F.text(1.6, 8.6, 'FFN 8d²', { size: 16, color: F.P.green, anchor: 'middle' });
F.text(2.8, 4.2, '每 Block\\n12d²', { size: 16, color: F.P.deep, anchor: 'middle' });
F.text(1.2, 1.4, 'FFN 占 2/3——宽比深便宜', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab35-pretraining", "train-val-divergence", "1.1", 320, "过拟合的分歧点",
    "train loss 持续下降、val loss 触底回升：小语料上大模型（$d=512$）的分歧点来得早——容量-数据的匹配问题。",
    "train/val 双曲线", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: 0, y1: 2.5, labels: ['step / 100', 'loss'] });
F.curve(t => 2.3 * Math.exp(-0.05 * t) + 0.35, { t0: 0, t1: 100, color: F.P.ink, width: 2.2, label: 'train', labelAt: 70 });
F.curve(t => 2.3 * Math.exp(-0.05 * t) + 0.5 + 0.9 / (1 + Math.exp(0.18 * (55 - t))), { t0: 0, t1: 100, color: F.P.green, width: 2.2, label: 'val', labelAt: 70 });
F.vbar(58, 2.4, { w: 0.8, fill: 'rgba(200,60,60,0.35)', color: 'rgba(0,0,0,0)', width: 0 });
F.text(58, 0.7, '分歧点', { size: 15, color: 'rgba(200,60,60,0.9)', anchor: 'middle' });
""")

fig("lab35-pretraining", "ppl-scale", "1.3", 300, "困惑度与规模",
    "PPL $= e^{\\mathcal L}$：交叉熵的指数——「等效均匀候选数」。模型规模的收益递减；上下文长度的收益在小语料上先饱和。",
    "PPL vs d", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 64, x1: 512, y0: 4, y1: 20, labels: ['d', 'PPL'] });
F.curve(d => 19 - 14 * Math.exp(-0.009 * (d - 64)), { t0: 64, t1: 512, color: F.P.ink, width: 2.3, label: '验证集', labelAt: 300 });
F.text(150, 7, 'PPL=10：每个位置等效 10 选 1', { size: 15, color: F.P.dim });
""")

fig("lab36-inference", "kv-cache", "1.2", 330, "KV Cache：省掉重复劳动",
    "因果注意力下位置 $t$ 只看 $\\le t$：历史 K/V 不变，缓存后每步只算新 token 的 $q, k, v$——$O(T^2 d) \\to O(Td)$。",
    "无缓存（全重算）vs 缓存（增量）", """
const F = Fig(document.getElementById('fig'));
for (const [x0, tag, col] of [[60, '无 cache', F.P.dim], [400, '有 cache', F.P.ink]]) {
  F.text(x0 + 130, 40, tag, { size: 18, color: col, anchor: 'middle' });
  for (let i = 0; i < 5; i++) {
    F.box({ x: x0 + i * 52, y: 70, w: 44, h: 40, label: i < 4 ? 'k' : 'k', fill: i === 4 ? 'rgba(191,135,0,0.25)' : 'rgba(58,74,168,0.12)', size: 13 });
    F.box({ x: x0 + i * 52, y: 120, w: 44, h: 40, label: 'v', fill: i === 4 ? 'rgba(191,135,0,0.25)' : 'rgba(58,74,168,0.12)', size: 13 });
  }
  if (tag === '无 cache') {
    for (let i = 0; i < 4; i++) { F.box({ x: x0 + i * 52, y: 70, w: 44, h: 40, fill: 'rgba(200,60,60,0.2)', label: '×', color: 'rgba(150,60,60,0.9)', size: 13 }); }
    F.text(x0 + 104, 190, '每步全部重算（琥珀+红叉）', { size: 14, color: F.P.dim, anchor: 'middle' });
  } else {
    F.text(x0 + 130, 190, '只算新 token（琥珀）', { size: 14, color: F.P.dim, anchor: 'middle' });
  }
}
F.text(360, 240, '延迟: 线性恶化 → 近乎平坦（512 长度差一个量级）', { size: 16, color: F.P.deep, anchor: 'middle' });
""")

fig("lab36-inference", "temperature-shapes", "1.1", 300, "温度与截断的分布形态",
    "$T \\to 0$ 退化为 argmax、$T$ 大趋于均匀（秩不变）；top-k 硬截断（固定候选数）、top-p 自适应（分布尖时少保留）。",
    "同分布不同 T", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 10, y0: 0, y1: 0.6, labels: ['token', '概率'] });
const logits = [0, 1, 2, 3, 3.4, 2.6, 1.4, 0.5, 0.1, 0];
for (const [T, col] of [[0.5, F.P.ink], [1.0, F.P.amber], [1.6, F.P.dim]]) {
  const exps = logits.map(l => Math.exp(l / T));
  const Z = exps.reduce((a, b) => a + b, 0);
  F.curve(t => {
    const k = Math.min(9, Math.max(0, Math.round(t)));
    return exps[k] / Z;
  }, { t0: 0, t1: 9, color: col, width: 2, label: 'T=' + T, labelAt: 0.4, rough: false });
}
""")

fig("lab37-sft", "prompt-mask", "1.2", 310, "SFT：只对回复计损",
    "prompt 段 label 全 $-100$（不学「生成指令」），损失归一化只除以 completion token 数——指令模板高度重复时不掩码会过拟合模板而非映射。",
    "序列的两段与掩码", """
const F = Fig(document.getElementById('fig'));
const segs = [['<指令>', 'rgba(0,0,0,0.06)', 150], ['<回复>', 'rgba(26,127,55,0.12)', 220]];
let x = 100;
for (const [tag, fill, w] of segs) {
  F.box({ x, y: 80, w, h: 56, label: tag, fill, size: 16 });
  F.text(x + w / 2, 170, 'label = ' + (tag.includes('指令') ? '−100（全掩）' : 'y_t（正常）'), { size: 15, color: tag.includes('指令') ? F.P.dim : F.P.green, anchor: 'middle' });
  x += w + 14;
}
F.text(360, 230, '分母 = |completion|（不是序列总长）', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab37-sft", "forgetting", "1.3", 290, "灾难性遗忘的量级",
    "SFT 后任务准确率跳升、通用 PPL 轻微上升——小数据全量微调的遗忘是可接受的代价；LoRA（Lab 38）把它进一步压小。",
    "SFT 前后的两个指标", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 3, y0: 0, y1: 1.1, labels: ['', '归一值'] });
F.vbar(0.7, 0.15, { w: 0.5, fill: 'rgba(58,74,168,0.2)' });
F.vbar(1.3, 0.88, { w: 0.5, fill: 'rgba(58,74,168,0.35)' });
F.text(1.0, 0.02, '任务准确率（SFT 前→后）', { size: 15, color: F.P.ink, anchor: 'middle' });
F.vbar(2.2, 0.5, { w: 0.5, fill: 'rgba(26,127,55,0.2)' });
F.vbar(2.8, 0.56, { w: 0.5, fill: 'rgba(26,127,55,0.35)' });
F.text(2.5, 0.02, '通用 PPL（升 ~10%）', { size: 15, color: F.P.green, anchor: 'middle' });
""")

fig("lab38-lora", "lora-structure", "1.1", 300, "LoRA：冻结主干 + 低秩旁路",
    "$h = W_0 x + \\frac{\\alpha}{r} B A x$，$B = 0$ 初始化（起点严格等于原模型）；$\\Delta W$ 的低秩假设——任务适配改变行为子空间而非全部表征。",
    "W₀ 冻结, B·A 可训练", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 60, y: 120, w: 170, h: 60, label: 'W₀（冻结）', fill: 'rgba(0,0,0,0.04)', color: F.P.dim, size: 16 });
F.box({ x: 60, y: 40, w: 64, h: 40, label: 'A (r×d)', fill: 'rgba(26,127,55,0.12)', color: F.P.green, size: 13 });
F.box({ x: 170, y: 40, w: 64, h: 40, label: 'B (d×r), =0', fill: 'rgba(26,127,55,0.12)', color: F.P.green, size: 12 });
F.arrow(128, 80, 172, 80, { color: F.P.green, width: 1.6, label: '×', labelDy: -10 });
F.path('M 96 120 L 96 80', { color: F.P.green, width: 1.4 });
F.path('M 202 80 L 202 120', { arrow: true, color: F.P.green, width: 1.4 });
F.text(300, 150, '+', { size: 28, color: F.P.ink });
F.text(340, 150, 'h = W₀x + (α/r)·BAx', { size: 16, color: F.P.deep, anchor: 'start' });
F.text(360, 220, 'r=8, d=256：可训练参数 6.25%', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab38-lora", "param-vs-quality", "1.3", 300, "低秩的饱和点",
    "r=8 已接近全量微调 95%+ 的任务效果；r 继续增大收益递减——Eckart-Young：谱衰减快时低秩逼近便宜。",
    "准确率 vs 可训练参数占比", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 26, y0: 0.5, y1: 1.0, labels: ['可训练参数 %（r=2/8/32）', '相对全量的效果'] });
for (const [r, xp] of [[2, 1.6], [8, 6.3], [32, 25]]) {
  F.vbar(xp, 0.72 + 0.012 * r, { w: 1.2, fill: 'rgba(58,74,168,0.25)' });
  F.text(xp, 0.56, 'r=' + r, { size: 15, color: F.P.deep, anchor: 'middle' });
}
F.curve(t => 0.95 + 0.03 / (1 + 8 / t), { t0: 1, t1: 25, color: F.P.green, width: 1.8, rough: false, label: '全量对照', labelAt: 14 });
""")

fig("lab39-dpo", "bt-sigmoid", "1.1", 320, "Bradley-Terry：偏好 = 奖励差的 sigmoid",
    "$p(y_w \\succ y_l) = \\sigma(r_w - r_l)$：偏好数据只有相对好坏，BT 把相对判断映射到标量差——平局（$r_w = r_l$）时损失 $\\ln 2$。",
    "σ(rw − rl) 与平局点", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -5, x1: 5, y0: 0, y1: 1.05, labels: ['r_w − r_l', 'P(y_w ≻ y_l)'] });
F.curve(t => 1 / (1 + Math.exp(-t)), { t0: -5, t1: 5, color: F.P.ink, width: 2.4, label: 'σ(Δr)', labelAt: -3.4 });
F.vbar(0, 0.5, { w: 0.03, fill: 'rgba(0,0,0,0.3)', color: 'rgba(0,0,0,0)', width: 0 });
F.text(0.3, 0.56, '平局: loss = ln2 ≈ 0.693', { size: 15, color: F.P.dim, anchor: 'start' });
""")

fig("lab39-dpo", "implicit-reward-shift", "1.3", 320, "DPO 的隐式奖励搬家",
    "训练前后 $\\hat r = \\beta \\log(\\pi/\\pi_{ref})$：$y_w$ 右移、$y_l$ 左移、可分性增强——损失直接作用在策略上，RM 与 RL 被折叠成一步。",
    "rw/rl 分布的训练前后", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -4, x1: 4, y0: 0, y1: 0.55, labels: ['隐式奖励 r̂', '密度'] });
const gauss = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * 2.51);
F.curve(x => gauss(x, -0.4, 1.1), { t0: -4, t1: 4, color: F.P.dim, width: 1.6, label: 'y_w 前', labelAt: -1.8 });
F.curve(x => gauss(x, 0.4, 1.1), { t0: -4, t1: 4, color: F.P.dim, width: 1.6, label: 'y_l 前', labelAt: 1.8 });
F.curve(x => gauss(x, 1.6, 0.7), { t0: -4, t1: 4, color: F.P.green, width: 2.2, label: 'y_w 后', labelAt: 2.8 });
F.curve(x => gauss(x, -1.6, 0.7), { t0: -4, t1: 4, color: F.P.ink, width: 2.2, label: 'y_l 后', labelAt: -2.8 });
""")

fig("lab40-reward-model", "rm-separation", "1.1", 310, "Reward Model 的判别",
    "RM 分数：$r_w$ 分布右移、$r_l$ 左移，判别 AUC 即 held-out 准确率；分不开的对贡献最大梯度（logistic 的难样本加权）。",
    "rw 与 rl 的分数分布", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -4, x1: 4, y0: 0, y1: 0.55, labels: ['RM 分数', '密度'] });
const gauss = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * 2.51);
F.curve(x => gauss(x, 1.3, 0.8), { t0: -4, t1: 4, color: F.P.green, width: 2.2, label: 'r(x, y_w)', labelAt: 2.6 });
F.curve(x => gauss(x, -1.3, 0.8), { t0: -4, t1: 4, color: F.P.ink, width: 2.2, label: 'r(x, y_l)', labelAt: -2.6 });
F.text(0, 0.5, '重叠区 = 难样本（梯度最大）', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab40-reward-model", "length-bias", "1.3", 290, "RM 的表面特征依赖",
    "同一回答 padding 加长后分数系统性漂移：RM 容易学到长度等表面特征——用 RM 当奖励的策略会学会刷这些特征（reward hacking 的入口）。",
    "分数随长度的漂移", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: 0, y1: 3, labels: ['log 长度', 'RM 分数'] });
F.curve(t => 1.4 + 0.55 * t, { t0: 0.1, t1: 3.9, color: F.P.ink, width: 2.3, label: '实测漂移', labelAt: 1.0 });
F.curve(t => 1.4, { t0: 0.1, t1: 3.9, color: F.P.green, width: 1.4, rough: false, label: '理想（应无关）', labelAt: 2.6 });
F.text(1.8, 0.6, '偏倚 = 未来被策略利用的空间', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab41-grpo", "group-advantage", "1.1", 330, "组内相对优势",
    "同 prompt 采 $G$ 个回答、按可验证判分：$\\hat A_i = (r_i - \\mathrm{mean})/\\mathrm{std}$——全对/全错组 $\\hat A \\equiv 0$（无信号，弃用）。",
    "组内奖励与归一化优势", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 8, y0: -1.6, y1: 1.6, labels: ['G 个回答', 'r / Â'] });
const rs = [1, 0, 1, 1, 0, 0, 1, 0];
const mean = rs.reduce((a, b) => a + b) / 8, sd = Math.sqrt(rs.reduce((a, b) => a + (b - mean) ** 2, 0) / 8);
for (let i = 0; i < 8; i++) {
  F.vbar(i + 0.32, rs[i], { w: 0.3, fill: 'rgba(58,74,168,0.3)' });
  F.vbar(i + 0.72, (rs[i] - mean) / sd, { w: 0.3, fill: 'rgba(26,127,55,0.3)' });
}
F.curve(t => 0, { t0: 0, t1: 8, color: 'rgba(0,0,0,0.2)', width: 0.8, rough: false });
F.text(0.55, 1.45, 'r（左蓝）', { size: 14, color: F.P.ink, anchor: 'middle' });
F.text(1.05, 1.45, 'Â（右绿）', { size: 14, color: F.P.green, anchor: 'middle' });
F.text(4, -1.45, 'mean=0.5, std=0.53 → Â = ±0.94', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab41-grpo", "signal-band", "1.3", 310, "学习带：隐式课程",
    "有信号组（组内既有对又有错）的占比随训练先升后降：太难（全错）与太简单（全对）的任务不产生更新——GRPO 的自动课程。",
    "有信号组比例的训练轨迹", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: 0, y1: 1.05, labels: ['训练进度', '有信号组占比'] });
F.curve(t => 0.55 * Math.exp(-((t - 45) ** 2) / 900) + 0.12, { t0: 0, t1: 100, color: F.P.ink, width: 2.3, label: '学习带', labelAt: 70 });
F.text(20, 0.25, '全错带（前期主导）', { size: 15, color: F.P.dim, anchor: 'middle' });
F.text(85, 0.25, '全对带（攻克后）', { size: 15, color: F.P.green, anchor: 'middle' });
""")

# ============================== stage 6 ==============================

fig("lab42-agent-loop", "agent-cycle", "1.1", 320, "Agent Loop：闭环",
    "观察 → 策略解码动作 → 工具执行 → 新观察……策略分布只定义在动作段上，观察段是注入的条件；终止动作或 $T_{max}$ 截断结束。",
    "obs → policy → action → tool 循环", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 250, y: 40, w: 200, h: 46, label: 'Observation', fill: 'rgba(58,74,168,0.08)' });
F.box({ x: 470, y: 130, w: 180, h: 46, label: 'π_θ 解码动作段', fill: 'rgba(26,127,55,0.08)' });
F.box({ x: 250, y: 220, w: 200, h: 46, label: 'Tool 执行', fill: 'rgba(191,135,0,0.1)' });
F.box({ x: 60, y: 130, w: 130, h: 46, label: '新观察拼接', fill: 'rgba(58,74,168,0.08)' });
F.arrow(450, 63, 540, 128, { width: 2 });
F.arrow(560, 176, 450, 240, { width: 2, label: 'action', labelDy: -12 });
F.arrow(250, 243, 125, 178, { width: 2, label: '返回结果', labelDy: 18 });
F.arrow(125, 128, 250, 63, { width: 2, label: '拼接进上下文', labelDy: 18 });
F.text(590, 100, 'env.step 的响应\\n任何乱码都有定义', { size: 13, color: F.P.dim, anchor: 'start' });
""")

fig("lab42-agent-loop", "tool-safety", "1.3", 290, "受限工具的边界",
    "白名单算符、禁 import/open/exec、超时兜底：环境的鲁棒性原则——任何模型输出都必须有定义好的响应，错误也是 observation。",
    "安全边界（进得来的才执行）", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 70, y: 60, w: 220, h: 46, label: 'py("import os; ...")', size: 14 });
F.box({ x: 70, y: 130, w: 220, h: 46, label: 'py("1+2*3")', size: 14 });
F.box({ x: 430, y: 90, w: 180, h: 46, label: '受限执行器', fill: 'rgba(26,127,55,0.08)' });
F.arrow(290, 83, 430, 100, { width: 1.8, color: 'rgba(200,60,60,0.8)', label: '拒绝', labelDy: -12 });
F.arrow(290, 153, 430, 130, { width: 2, color: F.P.green, label: '白名单', labelDy: 16 });
F.text(360, 215, '输出: INVALID / TIMEOUT / 结果——全是合法观察', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab43-trajectory", "failure-taxonomy", "1.3", 320, "失败分类树",
    "截断型（循环/无进展）与终止型（答错）的分野：协议型失败修 SFT、能力型失败交给 RL、空转型靠成本惩罚——训练前后按此对照。",
    "失败 → 归因 → 修法", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 280, y: 24, w: 140, h: 40, label: '失败', fill: 'rgba(200,60,60,0.08)' });
const kids = [
  ['协议型（大量 INVALID）', '修 SFT 数据', 20],
  ['循环型（同动作重复）', '成本惩罚 / 上下文困惑', 130],
  ['无进展（合法但空转）', '规划缺失 → RL', 250],
  ['答案错、工具链对', '算术/检索能力 → RL', 370],
];
for (const [tag, fix, x] of kids) {
  F.box({ x, y: 120, w: 190, h: 40, label: tag, size: 12, fill: 'rgba(58,74,168,0.06)' });
  F.box({ x, y: 200, w: 190, h: 40, label: fix, size: 12, fill: 'rgba(26,127,55,0.06)', color: F.P.green });
  F.arrow(340, 64, x + 95, 118, { width: 1.2 });
}
""")

fig("lab43-trajectory", "success-ci", "1.2", 290, "成功率与置信区间",
    "$n=200$ 时标准误 $\\approx \\sqrt{0.25/200} \\approx 0.035$：报告必须带 $\\pm 2\\mathrm{se}$——Lab 03 的区间规范在 Agentic 阶段的延续。",
    "成功率 + ±2se", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: 0, y1: 1, labels: ['温度', '成功率'] });
for (const [i, m] of [[0.6, 0.38], [1.6, 0.62], [2.6, 0.55], [3.6, 0.30]]) {
  F.vbar(i, m, { w: 0.5, fill: 'rgba(58,74,168,0.25)' });
  F.text(i, m + 0.07, (m * 100).toFixed(0) + '%', { size: 15, color: F.P.deep, anchor: 'middle' });
  F.vbar(i + 0.26, m + 0.07, { w: 0.02, fill: 'rgba(0,0,0,0.4)', color: 'rgba(0,0,0,0)', width: 0 });
}
F.text(2.1, 0.12, '±2se ≈ ±7%（n=200）', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab44-reward-design", "proxy-vs-true", "1.2", 320, "Reward Hacking 的标准图式",
    "代理指标上升、真实成功率停滞或下降——两者的差集就是被利用的空间：优化的是 $R$，从来不是 $R^*$。",
    "代理分 ↑ 而真实成功率 →", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: 0, y1: 1.05, labels: ['训练', '归一值'] });
F.curve(t => 1 - Math.exp(-0.04 * t), { t0: 0, t1: 100, color: F.P.ink, width: 2.3, label: '代理奖励 R', labelAt: 62 });
F.curve(t => 0.45 + 0.08 * Math.sin(t * 0.15) * Math.exp(-0.01 * t) - 0.12 / (1 + Math.exp(0.1 * (50 - t))), { t0: 0, t1: 100, color: F.P.green, width: 2.1, label: '真实成功率 R*', labelAt: 62 });
F.text(35, 0.25, '差集 = hack 的空间', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab44-reward-design", "weight-pareto", "1.3", 300, "成本惩罚的 Pareto 前沿",
    "$w_c$ 扫描：步数与成功率的交换曲线——过强的 $w_c$ 诱发「跳步瞎答」，过弱封不住空转刷分；前沿上的点才是可用配置。",
    "成功率 vs 平均步数（w_c 扫描）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 2, x1: 10, y0: 0.2, y1: 0.9, labels: ['平均步数', '成功率'] });
for (const [s, acc] of [[3, 0.3], [4.2, 0.55], [5.5, 0.72], [7, 0.78], [9, 0.74]]) {
  F.vbar(0, 0, { w: 0.01, fill: 'rgba(0,0,0,0)', color: 'rgba(0,0,0,0)', width: 0 });
  F.text(s, acc, '●', { size: 16, color: F.P.ink, anchor: 'middle' });
}
F.curve(t => 0.9 - 0.5 * Math.exp(-0.5 * (t - 3)), { t0: 3.2, t1: 8.8, color: F.P.amber, width: 1.8, rough: false, label: '前沿', labelAt: 6.5 });
F.text(9.2, 0.68, 'w_c 过强：\\n跳步乱答', { size: 13, color: F.P.dim, anchor: 'start' });
""")

fig("lab45-agent-rl", "seven-steps", "1.3", 300, "训练闭环的七步",
    "Prompt 采样 → Agent Rollout → 轨迹收集 → 奖励计算 → 组归一化优势 → token 级策略更新 → 评估——GRPO 迁到工具环境的全部增量在「轨迹是多步的」。",
    "on-policy 七步环", """
const F = Fig(document.getElementById('fig'));
const steps = ['Prompt', 'Rollout', '轨迹', 'Reward', 'Â 组内', 'Update', 'Eval'];
for (let i = 0; i < steps.length; i++) {
  F.box({ x: 40 + i * 92, y: 100, w: 78, h: 40, label: steps[i], size: 13, fill: 'rgba(58,74,168,0.07)' });
  if (i < 6) F.arrow(118 + i * 92, 120, 130 + i * 92, 120, { width: 1.6 });
}
F.path('M 40 120 C 10 120, 10 190, 120 190 L 600 190 C 670 190, 670 120, 640 145', { arrow: true, color: F.P.amber, width: 1.6, opacity: 0.6 });
F.text(360, 218, 'on-policy：数据用完即弃（多步下 off-policy 偏差被复合放大）', { size: 14, color: F.P.dim, anchor: 'middle' });
""")

fig("lab45-agent-rl", "before-after", "1.4", 300, "训练改变的是什么",
    "成功率跳升、无效动作与截断率大降、平均步数下降——协议型失败几乎不动：RL 修复行为质量，不修复格式遵循。",
    "训练前后指标对照", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: 0, y1: 1.05, labels: ['', '归一值'] });
const metrics = [['成功率', 0.35, 0.72, F.P.ink], ['无效动作率', 0.42, 0.12, 'rgba(200,60,60,0.8)'], ['截断率', 0.5, 0.18, F.P.dim], ['平均步数', 0.85, 0.55, F.P.amber]];
metrics.forEach(([tag, before, after, col], i) => {
  const x = 0.5 + i;
  F.vbar(x - 0.14, before, { w: 0.24, fill: 'rgba(0,0,0,0.18)' });
  F.vbar(x + 0.14, after, { w: 0.24, fill: 'rgba(58,74,168,0.3)' });
  F.text(x, 0.04, tag, { size: 13, color: col, anchor: 'middle' });
});
F.text(0.36, 0.98, '前', { size: 14, color: F.P.dim, anchor: 'end' });
F.text(0.64, 0.98, '后', { size: 14, color: F.P.ink, anchor: 'start' });
""")
