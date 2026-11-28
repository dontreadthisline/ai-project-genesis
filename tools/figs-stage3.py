# stage3 图定义（被 gen-figs.py exec 加载）

fig("lab15-vae", "vae-latent", "1.2", 340, "VAE 的二维潜空间",
    "训练后的 $q(z \\mid x)$ 簇按数字分离且大致填满先验 $\\mathcal N(0, I)$（虚线圆）；$\\beta$ 越大簇越向原点塌缩。",
    "潜空间簇 vs 标准正态先验", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3.4, x1: 3.4, y0: -2.8, y1: 2.8, labels: ['z₁', 'z₂'] });
F.ellipse({ rx: 2.3, ry: 2.3, color: F.P.dim, opacity: 0.5, width: 1.4 });
F.mtex(1.7, 2.4, '\\text{先验} N(0, I)', { size: 16, color: F.P.dim });
for (const [c, col] of [[-1.6, F.P.ink], [1.6, F.P.green], [0, F.P.amber]]) {
  for (let i = 0; i < 45; i++) {
    const a = F.rnd() * 6.28, r = Math.abs(F.gauss()) * 0.55;
    F.dot(0, 0, col, 2.1);  // noop占位防优化
  }
}
// 三簇: 中心分置, 手画点
const clus = [[-1.6, 0.9, F.P.ink], [1.7, 0.7, F.P.green], [0, -1.2, F.P.amber]];
for (const [cx, cy, col] of clus)
  for (let i = 0; i < 40; i++)
    F.text(cx + F.gauss() * 0.5, cy + F.gauss() * 0.4, '·', { size: 19, color: col });
F.text(-1.6, 1.7, '簇 A', { size: 16, color: F.P.deep, anchor: 'middle' });
F.text(1.7, 1.5, '簇 B', { size: 16, color: F.P.green, anchor: 'middle' });
""")

fig("lab15-vae", "elbo-tradeoff", "1.3", 310, "重构与 KL 的拉锯",
    "ELBO = 重构项 − KL 项：训练初期 KL 先被压低（$q$ 贴近先验），重构随后上升；$\\beta$ 加权改变平衡点（重构模糊 ↔ 潜空间规整）。",
    "两项的此消彼长", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 20, y0: 0, y1: 1.05, labels: ['epoch', '值（归一）'] });
F.curve(t => 1 / (1 + Math.exp(-0.7 * (t - 7))), { t0: 0, t1: 20, color: F.P.ink, width: 2.3, label: '重构项 ↑', labelAt: 4 });
F.curve(t => 1 / (1 + Math.exp(0.7 * (t - 4.5))), { t0: 0, t1: 20, color: F.P.amber, width: 2.1, label: 'KL 项 ↓', labelAt: 6 });
F.mtex(12.5, 0.35, '\\beta \\text{大} \\to KL \\text{压得更低、重构更糊}', { size: 15, color: F.P.dim });
""")

fig("lab16-reparameterization", "two-gradients", "1.1", 310, "两条梯度路径",
    "pathwise：梯度穿过 $\\mu + \\sigma\\epsilon$ 的确定性路径（局部信息）；score function：$f(z)\\nabla\\log q$（把 $f$ 的幅度当随机乘子——方差大）。",
    "pathwise vs score function", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 40, y: 120, w: 84, h: 46, label: 'θ' });
F.box({ x: 190, y: 120, w: 110, h: 46, label: 'μ, σ' });
F.box({ x: 360, y: 120, w: 90, h: 46, label: 'z = μ+σε', size: 14 });
F.box({ x: 530, y: 120, w: 90, h: 46, label: 'f(z)' });
F.arrow(124, 143, 190, 143, { width: 2, label: '梯度可传', labelDy: -12 });
F.arrow(300, 143, 360, 143, { width: 2 });
F.arrow(450, 143, 530, 143, { width: 2 });
F.text(360, 230, 'pathwise：∇f · ∇z —— 局部、低方差', { size: 16, color: F.P.ink, anchor: 'middle' });
F.mtex(360, 262, 'score \\text{：} f(z) ∇log q(z|\\theta) —— \\text{幅度进方差}', { size: 16, color: F.P.amber, anchor: 'middle' });
""")

fig("lab16-reparameterization", "variance-ratio", "1.2", 300, "两种估计的方差差",
    "同一目标 $\\nabla_\\mu \\mathbb E[f]$：score function 估计的样本方差随 $f$ 幅度平方增长，pathwise 不变——VAE 走 pathwise 的原因。",
    "估计量方差 vs f 的幅度", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: 0, y1: 16, labels: ['log |f| 幅度', 'log Var'] });
F.curve(t => 2 * t, { t0: 0.2, t1: 3.9, color: F.P.amber, width: 2.3, label: 'score（∝ f²）', labelAt: 1.1 });
F.curve(t => 1.2, { t0: 0.2, t1: 3.9, color: F.P.ink, width: 2.3, label: 'pathwise（不变）', labelAt: 2.2 });
""")

fig("lab17-gan", "d-optimal", "1.2", 330, "最优判别器与均衡",
    "$D^*(x) = p_{data}/(p_{data} + p_g)$（橙）在两分布重叠处取中间值；博弈均衡 $p_g = p_{data}$ 时 $D^* \\equiv 1/2$，$C(G) = -\\log 4$。",
    "p_data、p_g 与 D*", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -4, x1: 4, y0: 0, y1: 0.55, labels: ['x', '密度'] });
F.curve(t => Math.exp(-t * t / 2) / Math.sqrt(2 * Math.PI), { t0: -4, t1: 4, color: F.P.ink, width: 2.2, label: 'p_data', labelAt: -1.6 });
F.curve(t => Math.exp(-(((t - 1.8) ** 2)) / 1.4) / Math.sqrt(2 * Math.PI * 0.7), { t0: -4, t1: 4, color: F.P.green, width: 2, label: 'p_g', labelAt: 2.8 });
F.curve(t => {
  const a = Math.exp(-t * t / 2) / 2.5066, b = Math.exp(-(((t - 1.8) ** 2)) / 1.4) / 2.107;
  return a / (a + b) * 0.5;
}, { t0: -4, t1: 4, color: F.P.amber, width: 2.1, rough: false, label: 'D*(x)/2', labelAt: 0.6 });
F.mtex(-2.6, 0.44, '\\text{重叠区} D* \\approx 0.5', { size: 15, color: F.P.dim });
""")

fig("lab17-gan", "loss-oscillation", "1.4", 310, "GAN 训练的拉锯",
    "健康训练：D 准确率在 60%–90% 拉锯（谁也不能完胜）；任一条 loss 趋 0 即失衡——D 完胜则 G 无梯度，G 骗过一切则判别失效。",
    "D/G 损失的动态均衡", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 50, y0: 0, y1: 2.2, labels: ['step', 'loss'] });
F.curve(t => 1.1 + 0.35 * Math.sin(t * 0.7) * Math.exp(-0.01 * t), { t0: 0, t1: 50, color: F.P.ink, width: 2.1, label: 'D loss', labelAt: 38 });
F.curve(t => 1.05 - 0.3 * Math.sin(t * 0.7 + 0.5) * Math.exp(-0.01 * t), { t0: 0, t1: 50, color: F.P.green, width: 2.1, label: 'G loss', labelAt: 38 });
F.mtex(9, 2.0, '\\text{围绕均衡水平震荡} = \\text{健康}', { size: 16, color: F.P.dim });
""")

fig("lab18-flow", "flow-transform", "1.1", 330, "可逆变换的密度传输",
    "$z$ 空间标准正态圆云经可逆 $f$ 弯折成目标双峰；密度按 $|\\det J|$ 体积补偿——某方向压缩、密度必增大。",
    "圆云 → 双峰（体积补偿）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3.6, x1: 6.2, y0: -2.8, y1: 2.8, labels: ['', ''] });
for (let i = 0; i < 120; i++) F.text(-2 + F.gauss() * 0.8, F.gauss() * 0.8, '·', { size: 19, color: F.P.dim });
F.ellipse({ cx: -2, rx: 1.7, ry: 1.7, color: F.P.dim, opacity: 0.5, width: 1.4 });
F.mtex(-2, -2.3, 'z ~ N(0, I)', { size: 17, color: F.P.dim, anchor: 'middle' });
F.arrow(-0.1, 0, 2.2, 0, { label: 'f（可逆）', width: 2.2, labelDy: -14 });
for (let i = 0; i < 130; i++) {
  const b = F.rnd() < 0.5;
  F.text((b ? -1.9 : 1.9) + F.gauss() * 0.55, F.gauss() * 1.15, '·', { size: 19, color: F.P.ink });
}
F.ellipse({ cx: 4.5, rx: 1.9, ry: 1.35, color: F.P.ink, opacity: 0.6, width: 1.5 });
F.mtex(4.5, -2.3, 'x = f(z)', { size: 17, color: F.P.deep, anchor: 'middle' });
""")

fig("lab18-flow", "logdet-compensation", "1.2", 300, "Jacobian 行列式的补偿",
    "耦合层 $\\det J = \\exp(\\sum s)$：$s > 0$ 拉伸（密度稀释）、$s < 0$ 压缩（密度增浓）——对数似然里以 $-\\log|\\det|$ 记账。",
    "log|det| 随 Σs 的线性关系", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -2, x1: 2, y0: -4, y1: 4, labels: ['Σs', 'log|det J|'] });
F.curve(t => t, { t0: -2, t1: 2, color: F.P.ink, width: 2.3, label: 'log det = Σs', labelAt: 1.1 });
F.mtex(-1.2, 2.6, 's < 0 \\text{：压缩} \\n \\text{密度增大}', { size: 15, color: F.P.green, anchor: 'start' });
F.mtex(0.9, -2.9, 's > 0 \\text{：拉伸} \\n \\text{密度稀释}', { size: 15, color: F.P.dim, anchor: 'start' });
""")

fig("lab19-ddpm", "noise-chain", "1.1", 300, "前向加噪链",
    "$x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1 - \\bar\\alpha_t}\\, \\epsilon$：$\\bar\\alpha_t$ 从 1 衰减到 0，结构被噪声逐步淹没，$t = T$ 时是纯噪声。",
    "x₀ → x_T 的信号衰减", """
const F = Fig(document.getElementById('fig'));
for (let i = 0; i < 5; i++) {
  const a = 1 - i / 4.4;   // 信号占比
  const bx = 70 + i * 136;
  F.box({ x: bx, y: 90, w: 96, h: 96, label: '', fill: 'rgba(58,74,168,' + (0.05 + 0.55 * a).toFixed(2) + ')' });
  for (let k = 0; k < 22; k++)
    F.text(bx + 10 + F.rnd() * 76, 100 + F.rnd() * 76, '·', { size: 13, color: 'rgba(0,0,0,' + (0.5 * a + 0.08).toFixed(2) + ')' });
  F.mtex(bx + 48, 210, 't=' + (i * 75 || '0'), { size: 16, color: F.P.dim, anchor: 'middle' });
  if (i < 4) F.arrow(bx + 98, 138, bx + 132, 138, { width: 1.5 });
}
F.mtex(360, 250, 'ᾱt: 1 \\to 0 \\text{（信号} \\to \\text{噪声）}', { size: 17, color: F.P.dim, anchor: 'middle' });
""")

fig("lab19-ddpm", "loss-by-t", "1.2", 310, "按噪声水平的损失分布",
    "$\\epsilon$-预测让各 $t$ 的损失天然均匀（预测的是与 $t$ 无关的标准噪声）；$x_0$-预测在大 $t$ 损失巨大——凭空猜结构的代价。",
    "eps 预测 vs x₀ 预测的分桶损失", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 10, y0: 0, y1: 1.6, labels: ['t / T', 'loss'] });
for (let k = 0; k < 10; k++) F.vbar(k + 0.5, 0.45 + 0.08 * Math.sin(k), { w: 0.3, fill: 'rgba(58,74,168,0.25)' });
F.curve(t => 0.5 + 0.05 * Math.sin(4 * t), { t0: 0.3, t1: 9.7, color: F.P.ink, width: 2.2, label: 'ε 预测：均匀', labelAt: 2 });
F.curve(t => 0.15 + 0.13 * t * t / 10, { t0: 0.3, t1: 9.7, color: F.P.dim, width: 1.8, label: 'x₀ 预测：大 t 爆炸', labelAt: 3.4 });
""")

fig("lab20-flow-matching-2d", "vector-field", "1.1", 330, "速度场：噪声指向数据",
    "学到的 $v_\\theta(x, t)$：噪声区域的箭头指向数据高密区，数据区内近零——ODE 积分 = 沿箭头走，质量从 $\\mathcal N(0, I)$ 流向 $p_{data}$。",
    "速度场把噪声导向数据", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3.2, x1: 3.2, y0: -2.6, y1: 2.6, labels: ['', ''] });
for (let i = -2; i <= 2; i++) for (let j = -1; j <= 1; j++) {
  const x = i * 1.15, y = j * 1.15;
  const dx = (x > 0.9 ? -0.5 : 0.55), dy = (y > 0.6 ? -0.4 : 0.45);
  const len = Math.hypot(dx, dy);
  F.arrow((x - 3.4) * 90 + 360, (2.6 - y) * 55 + 165, (x - 3.4) * 90 + 360 + dx * 34, (2.6 - y) * 55 + 165 - dy * 34, { width: 1.4 });
}
for (let k = 0; k < 40; k++) F.text(3.6 + F.gauss() * 0.7, 0.4 + F.gauss() * 0.9, '·', { size: 19, color: F.P.ink });
F.text(3.5, 2.2, '数据簇', { size: 17, color: F.P.deep });
F.text(-2.6, 2.2, '噪声区：箭头指向数据', { size: 15, color: F.P.dim });
""")

fig("lab20-flow-matching-2d", "ode-steps", "1.3", 300, "积分步数与轨迹弯曲度",
    "直线路径学得准时少步即可（欧拉 1 步近似）；两簇之间平均方向打架的区域轨迹弯曲——步数不足则生成样本落在过渡带。",
    "NFE 少 → 轨迹须近乎直线", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 6, y0: 0, y1: 1, labels: ['ODE 步数', '生成质量'] });
F.curve(t => 1 - Math.exp(-0.9 * t), { t0: 0.2, t1: 6, color: F.P.ink, width: 2.3, label: '1 − e^{−0.9·NFE}', labelAt: 2.2 });
F.curve(t => 0.9, { t0: 0.2, t1: 6, color: F.P.green, width: 1.3, rough: false, label: '实用阈', labelAt: 4.4 });
F.mtex(0.6, 0.42, '5-8 \\text{步即可用}', { size: 16, color: F.P.dim });
""")

fig("lab21-flow-matching-mnist", "time-embedding", "1.2", 300, "时间条件化",
    "同一网络对不同 $t$ 输出不同速度（正弦嵌入 + FiLM 注入）——没有 $t$ 条件，一个网络无法表达整条概率路径的时变场。",
    "v(x, t) 随 t 改变", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 1, y0: -1.6, y1: 1.6, labels: ['t', 'v(x₀, t)'] });
for (const [k, col] of [[1, F.P.ink], [3, F.P.amber], [6, F.P.dim]]) {
  F.curve(t => Math.sin(k * Math.PI * t) * (1.3 - 0.15 * k / 6), { t0: 0, t1: 1, color: col, width: 1.8, label: 't 嵌入维度组 ' + k, labelAt: 0.06 });
}
""")

fig("lab21-flow-matching-mnist", "nfe-quality", "1.3", 300, "NFE-质量前沿",
    "4 步已可辨数字（直线路径的收益）；64 步细节干净——计算与质量的交换曲线，与解码加速（Lab 36）同属一主题。",
    "质量 vs 积分步数", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 64, y0: 0.3, y1: 1.0, labels: ['NFE', '质量（对高步参照）'] });
F.curve(t => 0.42 + 0.55 / (1 + 8 / t), { t0: 1, t1: 64, color: F.P.ink, width: 2.3, label: 'FM', labelAt: 26 });
F.vbar(4, 0.62, { w: 0.7, fill: 'rgba(58,74,168,0.3)' });
F.text(4.4, 0.5, '4 步可辨', { size: 15, color: F.P.dim, anchor: 'start' });
""")

fig("lab22-diffusion-vs-fm", "unified-path", "1.1", 320, "统一插值视角",
    "$x_t = a(t) x_1 + b(t) x_0$：DDPM 的方差保持调度与 FM 的直线路径是同族——目标只差与 $(t, x_t)$ 相关的确定性重缩放。",
    "两种调度下的 a(t), b(t)", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 1, y0: 0, y1: 1.05, labels: ['t', '系数'] });
F.curve(t => t, { t0: 0, t1: 1, color: F.P.ink, width: 2.3, label: 'FM: a(t)=t', labelAt: 0.18 });
F.curve(t => 1 - t, { t0: 0, t1: 1, color: F.P.green, width: 1.8, label: 'FM: b(t)=1−t', labelAt: 0.5 });
F.curve(t => Math.sqrt(1 - Math.exp(-3 * t * t)), { t0: 0, t1: 1, color: F.P.dim, width: 1.8, label: 'DDPM: √ᾱ(t)', labelAt: 0.55 });
F.curve(t => Math.sqrt(Math.exp(-3 * t * t)), { t0: 0, t1: 1, color: F.P.amber, width: 1.8, label: 'DDPM: √(1−ᾱ)', labelAt: 0.42 });
""")

fig("lab22-diffusion-vs-fm", "nfe-frontier", "1.1", 300, "少步区间的分野",
    "NFE ≤ 8 的少步区间 FM 通常占优（路径近直）；高步数两端趋同——两方法本质是同一族的不同调度。",
    "质量-NFE 双曲线", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 128, y0: 0.4, y1: 1.0, labels: ['NFE', '质量'] });
F.curve(t => 0.5 + 0.47 / (1 + 5 / t), { t0: 1, t1: 128, color: F.P.ink, width: 2.3, label: 'FM', labelAt: 40 });
F.curve(t => 0.46 + 0.5 / (1 + 22 / t), { t0: 1, t1: 128, color: F.P.dim, width: 1.9, label: 'DDIM', labelAt: 60 });
F.vbar(8, 0.97, { w: 0.6, fill: 'rgba(0,0,0,0.18)', color: 'rgba(0,0,0,0)', width: 0 });
F.text(2, 0.86, '少步区', { size: 15, color: F.P.dim });
""")
