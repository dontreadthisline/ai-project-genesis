# stage2 图定义（被 gen-figs.py exec 加载，fig/FIGS 来自其命名空间）

fig("lab06-mlp-from-scratch", "net-structure", "1.1", 300, "MLP：全连接的层级结构",
    "全连接：每对相邻神经元都有权重。示意 $5 \\to 4 \\to 3 \\to 2$（实作 $784 \\to 256 \\to 128 \\to 10$），参数量 $\\sum d_l d_{l+1}$。",
    "全连接网络（压缩维度示意）", """
const F = Fig(document.getElementById('fig'));
const L = [5, 4, 3, 2], X = [110, 300, 490, 640];
const ys = L.map((n, i) => F.layer({ x: X[i], y: 150, n, spread: 52, label: i ? '' : '输入' }));
for (let i = 0; i < 3; i++) F.links(ys[i], ys[i + 1], X[i] + 8, X[i + 1] - 8, { opacity: 0.22 });
F.text(300, 278, '隐藏层（ReLU）', { size: 16, color: F.P.dim, anchor: 'middle' });
F.text(640, 278, '输出', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab06-mlp-from-scratch", "grad-flow", "1.1", 300, "反向传播的梯度流",
    "反向传播 = 逆序 VJP：上游梯度 $\\bar y = W^T \\bar{out}$ 逐层回传。He 初始化下梯度范数同量级衰减、不塌缩。",
    "逐层梯度范数（输出→输入）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -0.5, x1: 4.5, y0: 0, y1: 1.2, labels: ['层', '||grad||'] });
for (let l = 0; l <= 4; l++) F.vbar(l, 1.0 - 0.13 * l, { w: 0.42, fill: 'rgba(58,74,168,0.25)' });
F.curve(t => 1.0 - 0.13 * t, { t0: 0, t1: 4, color: F.P.amber, width: 1.6, rough: false });
F.text(2, 0.25, '同量级衰减（He 初始化的功劳）', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab07-optimizers", "optimizer-curves", "1.2", 330, "三个优化器的收敛形态",
    "Adam 族早期下降最快（自适应步长）；调好学习率的 Momentum 终精度相当——SGD 需更小 $\\eta$ 与更长耐心。",
    "优化器 loss 曲线（形态示意）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 50, y0: 0, y1: 2.4, labels: ['epoch', 'loss'] });
F.curve(t => 2.3 / (1 + 0.28 * t) + 0.12, { t0: 0, t1: 50, color: F.P.dim, label: 'SGD', labelAt: 32 });
F.curve(t => 2.3 / (1 + 0.65 * t) + 0.05, { t0: 0, t1: 50, color: F.P.ink, width: 2.3, label: 'Momentum', labelAt: 24 });
F.curve(t => 2.3 / (1 + 1.5 * t) + 0.03, { t0: 0, t1: 50, color: F.P.amber, width: 2.3, label: 'Adam', labelAt: 9 });
F.mtex(2, 0.35, '\\text{初始} \\approx ln 10 \\approx 2.30 \\text{（随机预测）}', { size: 15, color: F.P.dim });
""")

fig("lab07-optimizers", "adam-bias-fix", "1.2", 310, "Adam 偏差修正",
    "前几步 $m_t$ 是 $g(1 - \\beta_1^t)$ 的有偏估计；$\\hat m = m/(1-\\beta_1^t)$ 修正后，常数梯度下有效步长立即达到 $\\eta$。",
    "m 有偏 vs m̂ 修正", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 40, y0: 0, y1: 1.15, labels: ['t', 'm / g'] });
F.curve(t => 1 - Math.pow(0.9, t), { t0: 1, t1: 40, color: F.P.ink, width: 2.2, label: 'm_t（有偏）', labelAt: 14 });
F.curve(t => 1, { t0: 1, t1: 40, color: F.P.amber, width: 1.8, rough: false, label: 'm̂ 修正后', labelAt: 24 });
F.mtex(3.5, 0.42, 't=1 \\text{时} m \\text{只有} 0.1\\cdotg', { size: 15, color: F.P.dim });
""")

fig("lab08-batch-size", "variance-scaling", "1.1", 320, "梯度噪声随 batch 的缩放",
    "batch 梯度方差 $\\propto 1/b$（斜率 $-1$）、信噪比 $\\propto \\sqrt{b}$（斜率 $+1/2$）——Lab 03 的 Monte Carlo 结构在优化中的重现。",
    "Var ∝ 1/b 与 SNR ∝ √b", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: -2, y1: 1, labels: ['log b', 'log 量'] });
F.curve(t => -t, { t0: 0.1, t1: 3.9, color: F.P.ink, width: 2.2, label: 'log Var（−1）', labelAt: 1.15 });
F.curve(t => 0.5 * t - 1.7, { t0: 0.1, t1: 3.9, color: F.P.amber, width: 2.2, label: 'log SNR（+1/2）', labelAt: 1.6 });
for (let i = 0; i < 10; i++) {
  const t = 0.2 + F.rnd() * 3.5;
  F.text(t, -t + (F.rnd() - 0.5) * 0.3, '·', { size: 20, color: F.P.ink });
}
""")

fig("lab08-batch-size", "cos-vs-b", "1.2", 310, "梯度方向的信噪比",
    "batch 梯度与全量梯度的余弦相似度随 $b$ 上升；$\\cos > 0.9$ 后方向收益递减——大 batch 的真实收益在吞吐不在方向。",
    "cos(batch, full) vs b", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 4096, y0: 0.4, y1: 1.0, labels: ['b', 'cos'] });
F.curve(b => 1 - 0.5 / Math.sqrt(b), { t0: 1, t1: 4096, color: F.P.ink, width: 2.2, label: '典型曲线', labelAt: 500 });
F.curve(b => 0.9, { t0: 1, t1: 4096, color: F.P.green, width: 1.4, rough: false, label: '0.9 实用阈', labelAt: 1600 });
""")

fig("lab09-cnn-vs-mlp", "conv-window", "1.1", 330, "卷积：局部窗口与参数共享",
    "同一 kernel 在全部位置复用（参数共享）；输出尺寸 $m = \\lfloor (n + 2p - K)/s \\rfloor + 1$。",
    "kernel 滑动（同一组权重）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 8.4, y0: 0, y1: 6.4, labels: ['', ''], pad: [16, 16, 16, 16], bare: true });
F.heat([[0.2,0.5,0.9,0.6,0.3,0.1],[0.3,0.8,1.0,0.7,0.4,0.2],[0.1,0.6,0.9,0.8,0.5,0.3],
        [0.4,0.7,0.8,0.6,0.3,0.2],[0.2,0.5,0.6,0.4,0.2,0.1],[0.1,0.3,0.4,0.3,0.1,0.05]],
       { x: 0.5, y: 0.55, cw: 1.15, ch: 1.15 });
const bx = t => 56 + (t + 0.0) * 86;
F.box({ x: bx(1.95), y: 92, w: 84, h: 84, color: F.P.amber, fill: 'rgba(0,0,0,0)', width: 2.2 });
F.arrow(bx(2.9) + 50, 134, bx(3.5) + 10, 134, { label: '滑动', width: 1.8, labelDy: -12 });
F.mtex(360, 66, '\\text{同一个} 3\\times3 kernel', { size: 18, color: F.P.deep, anchor: 'start' });
F.text(360, 96, '在所有位置复用', { size: 18, color: F.P.deep, anchor: 'start' });
F.box({ x: 360, y: 124, w: 76, h: 76, color: F.P.ink });
""")

fig("lab09-cnn-vs-mlp", "receptive-field", "1.3", 300, "感受野的线性增长",
    "两层 $3\\times3$（$RF=5$，参数 $18C^2$）等价一层 $5\\times5$（$25C^2$）的感受野——小核堆叠以更少参数换更多非线性。",
    "RF = 1 + 2L", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 6, y0: 0, y1: 14, labels: ['层数 L', '感受野'] });
F.curve(t => 1 + 2 * t, { t0: 1, t1: 6, color: F.P.ink, width: 2.2, label: '3×3 堆叠', labelAt: 1.7 });
for (let l = 1; l <= 6; l++) F.vbar(l, 1 + 2 * l, { w: 0.18, fill: 'rgba(58,74,168,0.2)' });
F.mtex(2.3, 11.8, 'L=4 \\to RF=9 \\text{（覆盖笔画结构）}', { size: 16, color: F.P.dim });
""")

fig("lab10-inductive-bias", "pixel-permute", "1.1", 300, "像素打乱：先验的证据",
    "固定置换 $\\Pi$ 打乱像素：MLP 可重参数化吸收（假设空间不变），CNN 的局部性先验彻底失效——同一数据、两种命运。",
    "置换破坏空间局部性", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 13, y0: 0, y1: 5.8, labels: ['', ''], pad: [16, 16, 16, 16], bare: true });
F.heat([[0.1,0.4,0.9,0.8],[0.3,0.8,1.0,0.5],[0.9,0.6,0.7,0.2],[0.2,0.5,0.3,0.1]], { x: 0.3, y: 0.2, cw: 1.4, ch: 1.4 });
F.heat([[0.9,0.2,0.3,0.1],[0.5,1.0,0.7,0.4],[0.1,0.8,0.2,0.9],[0.8,0.6,0.4,0.3]], { x: 7.4, y: 0.2, cw: 1.4, ch: 1.4 });
F.text(140, 288, '原图', { size: 18, color: F.P.deep, anchor: 'middle' });
F.text(580, 288, '同一组像素随机排列', { size: 18, color: F.P.deep, anchor: 'middle' });
F.arrow(300, 138, 440, 138, { label: 'Π', width: 2.2, labelDy: -12 });
""")

fig("lab10-inductive-bias", "acc-vs-n", "1.2", 320, "归纳偏置 = 小样本收益",
    "小数据时 CNN 的先验值钱（百级样本仍可用）；数据增大后先验被数据替代、差距收窄——偏置-方差互换的直接观测。",
    "准确率 vs 训练样本数", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 2, x1: 5, y0: 0.3, y1: 1.0, labels: ['log n', 'acc'] });
F.curve(t => 0.34 + 0.6 / (1 + Math.exp(-2.6 * (t - 3.4))), { t0: 2, t1: 5, color: F.P.ink, width: 2.3, label: 'CNN', labelAt: 2.6 });
F.curve(t => 0.25 + 0.68 / (1 + Math.exp(-3.4 * (t - 4.4))), { t0: 2, t1: 5, color: F.P.dim, width: 1.8, label: 'MLP', labelAt: 2.6 });
F.mtex(2.05, 0.9, 'n=100 \\text{：差距最大}', { size: 15, color: F.P.dim, anchor: 'start' });
""")

fig("lab11-residual", "grad-highway", "1.2", 320, "残差连接的梯度高速公路",
    "普通深网梯度 $\\prod_l J_l$ 指数衰减；残差块 $I + J_F$ 展开后恒等项永存——梯度下界不再乘性塌缩。",
    "首层梯度范数 vs 深度", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 2, x1: 32, y0: 0.02, y1: 1, labels: ['深度 L', '||∂L/∂h₀||'] });
F.curve(t => Math.pow(0.93, t), { t0: 2, t1: 32, color: F.P.dim, width: 1.9, label: 'plain：0.93^L', labelAt: 12 });
F.curve(t => 0.9 / (1 + 0.02 * t), { t0: 2, t1: 32, color: F.P.ink, width: 2.3, label: 'res：恒等项保底', labelAt: 14 });
F.mtex(6.5, 0.42, 'L=32 \\text{时} plain \\text{只剩} 10%', { size: 16, color: F.P.dim });
""")

fig("lab11-residual", "residual-block", "1.2", 300, "残差块：学习对恒等的修正",
    "零初始化 $F$ 末端时初始网络严格等于恒等映射——从「什么都不做」出发逐步增加修正（与 LoRA 的 $B=0$ 起步同构）。",
    "y = x + F(x)", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 60, y: 120, w: 90, h: 46, label: 'x' });
F.arrow(150, 143, 250, 143, { width: 1.8 });
F.box({ x: 250, y: 106, w: 150, h: 74, label: 'F(x)  末层零初始化', fill: 'rgba(26,127,55,0.07)', color: F.P.green, size: 15 });
F.path('M 150 143 C 200 55, 340 55, 430 132', { arrow: true, color: F.P.amber, width: 2 });
F.text(290, 60, '恒等捷径', { size: 17, color: F.P.amber });
F.mtex(214, 130, '+', { size: 26, color: F.P.ink });
F.box({ x: 430, y: 120, w: 120, h: 46, label: 'y = x + F(x)', size: 15 });
""")

fig("lab12-norm-reg", "bn-vs-ln", "1.1", 310, "BatchNorm 与 LayerNorm 的轴",
    "BN 对每个特征列（跨样本）标准化，LN 对每个样本（跨特征）——Transformer 必须用 LN：变长序列 + 自回归逐步解码，batch/时间轴统计都不可行。",
    "BN 按列 vs LN 按行", """
const F = Fig(document.getElementById('fig'));
const rows = 5, cols = 6, x0 = 70, y0 = 58, cw = 62, ch = 40;
for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
  F.box({ x: x0 + j * cw, y: y0 + i * ch, w: cw - 5, h: ch - 5, label: '', fill: 'rgba(58,74,168,0.04)', width: 0.7 });
  F.box({ x: x0 + 360 + j * cw, y: y0 + i * ch, w: cw - 5, h: ch - 5, label: '', fill: 'rgba(58,74,168,0.04)', width: 0.7 });
}
for (let i = 0; i < rows; i++)
  F.box({ x: x0 + 2 * cw, y: y0 + i * ch, w: cw - 5, h: ch - 5, fill: 'rgba(58,74,168,0.22)', color: F.P.ink, width: 1.3 });
for (let j = 0; j < cols; j++)
  F.box({ x: x0 + 360 + j * cw, y: y0 + 2 * ch, w: cw - 5, h: ch - 5, fill: 'rgba(191,135,0,0.2)', color: F.P.amber, width: 1.3 });
F.text(x0 + 2.5 * cw, y0 - 18, 'BN：按列（跨样本）', { size: 17, color: F.P.ink, anchor: 'middle' });
F.text(x0 + 360 + 3 * cw, y0 - 18, 'LN：按行（跨特征）', { size: 17, color: F.P.amber, anchor: 'middle' });
""")

fig("lab12-norm-reg", "dropout-mask", "1.3", 290, "Inverted Dropout",
    "训练时以概率 $p$ 置零并除以 $1-p$：期望不变、推理无需缩放——$2^n$ 个子网共享参数的隐式集成。",
    "mask 采样与期望保持", """
const F = Fig(document.getElementById('fig'));
const n = 8, x0 = 110, y0 = 52, cw = 56, ch = 46;
for (let i = 0; i < 2; i++) for (let j = 0; j < n; j++) {
  const drop = F.rnd() < 0.4;
  F.box({ x: x0 + j * cw, y: y0 + i * ch, w: cw - 6, h: ch - 6, label: drop ? '0' : '×1.67',
         fill: drop ? 'rgba(0,0,0,0.03)' : 'rgba(26,127,55,0.1)',
         color: drop ? F.P.dim : F.P.green, width: 1.1, size: 14 });
}
F.mtex(x0 + 3.5 * cw, y0 + 2.7 * ch, '\\text{两次前向} = \\text{两个子网（共享参数）}', { size: 17, color: F.P.dim, anchor: 'middle' });
""")

fig("lab13-rnn-lstm", "bptt-decay", "1.1", 320, "BPTT 的指数衰减",
    "$\\partial L/\\partial h_0 = \\prod_t W_h^T \\mathrm{diag}(1-h_t^2)$：谱半径 $\\rho < 1$ 时按 $\\rho^T$ 消失——50 步后只剩约 $10^{-10}$。",
    "梯度范数 vs 序列长度", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 50, y0: -22, y1: 0, labels: ['T', 'log ||∂L/∂h₀||'] });
for (const rho of [0.5, 0.85]) {
  const k = Math.log(rho);
  F.curve(t => 0.1 + k * t, { t0: 1, t1: 50, color: rho < 0.6 ? F.P.dim : F.P.ink, width: 2.1,
                              label: 'ρ = ' + rho, labelAt: 7 });
}
F.text(24, -13, 'tanh 饱和进一步加速衰减', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab13-rnn-lstm", "lstm-cell", "1.2", 320, "LSTM：加性细胞状态",
    "$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t$——对 $c$ 的梯度是 $\\mathrm{diag}(f_t)$（可学习趋近 1），不再是连乘 $W_h$。",
    "细胞状态的加性更新", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 70, y: 130, w: 90, h: 46, label: 'c_{t−1}' });
F.box({ x: 300, y: 130, w: 80, h: 46, label: 'c_t' });
F.arrow(160, 153, 300, 153, { width: 2.2, label: '× f_t', labelDy: -14 });
F.box({ x: 170, y: 58, w: 112, h: 44, label: 'i_t ⊙ c̃_t', fill: 'rgba(26,127,55,0.08)', color: F.P.green });
F.arrow(226, 102, 226, 130, { color: F.P.green, width: 1.8, label: '+', labelDy: -10 });
F.box({ x: 170, y: 212, w: 112, h: 44, label: 'f_t = σ(·)', fill: 'rgba(191,135,0,0.08)', color: F.P.amber });
F.arrow(226, 212, 226, 182, { color: F.P.amber, width: 1.6 });
F.mtex(400, 153, 'h_t = o_t \\odot tanh(c_t) \\to', { size: 16, color: F.P.deep, anchor: 'start' });
""")

fig("lab14-attention-transformer", "causal-mask", "1.1", 310, "因果掩码：时间箭头",
    "logits 加 $M_{ij}$（$j > i$ 取 $-\\infty$）：softmax 后未来权重严格为 0，一条序列同时提供 $T$ 个训练样本而无泄漏。",
    "下三角注意力掩码", """
const F = Fig(document.getElementById('fig'));
const n = 6, x0 = 160, y0 = 42, cw = 52, ch = 44;
for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
  const v = j <= i ? 0.35 + 0.6 * (i - j) / n : 0.03;
  F.box({ x: x0 + j * cw, y: y0 + i * ch, w: cw - 4, h: ch - 4, label: '',
         fill: 'rgba(58,74,168,' + (0.06 + 0.75 * v).toFixed(2) + ')', width: 0.6 });
}
F.mtex(x0 + n * cw / 2, y0 + n * ch + 26, '\\text{列} j = \\text{键（被看）；行} i = \\text{查询（看谁）}', { size: 16, color: F.P.dim, anchor: 'middle' });
F.mtex(x0 - 110, y0 + n * ch / 2, '\\text{上三角} = 0 \\text{：未来不可见}', { size: 15, color: F.P.dim });
""")

fig("lab14-attention-transformer", "mha-flow", "1.2", 300, "多头注意力的数据流",
    "$Q, K, V$ 各自投影到 $h$ 个子空间并行计算，concat 后经 $W^O$ 融合——单头只能强调一种模式，多头各看各的。",
    "Q/K/V → heads → concat → W^O", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 50, y: 130, w: 64, h: 44, label: 'Q' });
F.box({ x: 50, y: 62, w: 64, h: 44, label: 'K' });
F.box({ x: 50, y: 198, w: 64, h: 44, label: 'V' });
for (let i = 0; i < 4; i++) {
  F.box({ x: 210, y: 42 + i * 62, w: 116, h: 46, label: 'head ' + (i + 1), fill: 'rgba(58,74,168,0.08)' });
  F.arrow(114, 84 + i * 18, 210, 65 + i * 62, { width: 1.1 });
}
F.box({ x: 410, y: 130, w: 100, h: 44, label: 'concat' });
F.box({ x: 570, y: 130, w: 76, h: 44, label: 'W^O' });
F.arrow(326, 152, 410, 152, { width: 1.8 });
F.arrow(510, 152, 570, 152, { width: 1.8 });
F.mtex(360, 262, '\\sqrtd_k \\text{缩放} + softmax = \\text{各自的加权平均}', { size: 16, color: F.P.dim, anchor: 'middle' });
""")
