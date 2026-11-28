# stage4 图定义（被 gen-figs.py exec 加载）

fig("lab23-frozenlake", "gridworld", "1.1", 320, "FrozenLake：MDP 的最小实例",
    "$\\mathcal S = 16$ 个状态（含洞与目标的终止态），动作 4 个；slippery 下预期方向只以 $1/3$ 概率成立——转移核是环境的属性。",
    "4×4 网格世界（S/H/G 标注）", """
const F = Fig(document.getElementById('fig'));
const lab = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']];
const n = 4, x0 = 170, y0 = 30, cw = 84, ch = 62;
for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
  const t = lab[i][j];
  const fill = t === 'H' ? 'rgba(0,0,0,0.32)' : t === 'G' ? 'rgba(26,127,55,0.25)' : t === 'S' ? 'rgba(58,74,168,0.18)' : 'rgba(58,74,168,0.05)';
  F.box({ x: x0 + j * cw, y: y0 + i * ch, w: cw - 6, h: ch - 6, label: t, fill, width: 1.2, size: 18 });
}
F.text(90, y0 + n * ch / 2, 'S 起点 / H 洞（终止）\\nG 目标（终止）', { size: 15, color: F.P.dim });
""")

fig("lab23-frozenlake", "slippery", "1.2", 300, "slippery：动作 ≠ 结果",
    "预期「下」时三个实际转移各 $1/3$（含撞墙折叠回原地）——策略优化的对象是这个转移核，不是意图。",
    "预期方向与实际转移", """
const F = Fig(document.getElementById('fig'));
const cx = 220, cy = 170;
F.box({ x: cx - 50, y: cy - 46, w: 100, h: 92, label: 's', fill: 'rgba(58,74,168,0.1)' });
F.arrow(cx, cy + 52, cx, cy + 118, { label: '意图: 下', width: 2.4, labelDy: 18 });
F.arrow(cx + 60, cy + 30, cx + 60, cy + 96, { label: '下 1/3', width: 1.6, color: F.P.dim, labelDy: 16 });
F.arrow(cx + 130, cy - 10, cx + 130, cy - 74, { label: '左 1/3', width: 1.6, color: F.P.dim, labelDy: -10 });
F.arrow(cx + 200, cy - 10, cx + 200, cy - 74, { label: '右 1/3', width: 1.6, color: F.P.dim, labelDy: -10 });
F.mtex(cx + 230, cy + 60, 'P(s\\'|s,a)：环境说了算', { size: 16, color: F.P.deep, anchor: 'start' });
""")


fig("lab24-dynamic-programming", "value-landscape", "1.3", 320, "值函数的地形",
    "$v^*(s) \\approx \\gamma^{d(s)}$：离目标越远值越低（折扣的几何衰减），洞附近的梯度被 slippery 摊平——DP 迭代收敛后的地图。",
    "v*(s) 热图（目标在右下）", """
const F = Fig(document.getElementById('fig'));
const g = [[0.1,0.16,0.24,0.36],[0.07,0,0.33,0],[0.11,0.2,0.45,0],[0,0.7,0.85,1.0]];
F.axis({ x0: 0, x1: 5.4, y0: 0, y1: 4.4, labels: ['', ''], pad: [24, 24, 24, 24], bare: true });
F.heat(g, { x: 0.25, y: 0.25, cw: 1.2, ch: 1.02 });
F.mtex(4.85, 3.95, 'G', { size: 17, color: '#1a7f37' });
F.mtex(0.6, 0.6, 'S', { size: 17, color: '#2c3a8c' });
F.mtex(2.7, 4.15, 'v* \\approx \\gamma^d \\text{（距目标步数} d \\text{）}', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab24-dynamic-programming", "vi-convergence", "1.3", 300, "值迭代的几何收敛",
    "$T^*$ 是 $\\gamma$-压缩：误差上界 $\\|v_k - v^*\\| \\le \\gamma^k/(1-\\gamma)\\|v_1 - v_0\\|$——$\\gamma$ 越大越准但收敛越慢。",
    "误差 vs 迭代数（按 γ）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 30, y0: -6, y1: 0, labels: ['迭代 k', 'log 误差'] });
for (const [g, col] of [[0.5, F.P.dim], [0.9, F.P.ink]]) {
  F.curve(t => -2 + t * Math.log(g), { t0: 0, t1: Math.min(30, 46 * Math.abs(Math.log(0.9) / Math.log(g)) * 0.6),
             color: col, width: 2.2, label: 'γ=' + g, labelAt: 4 });
}
F.mtex(16, -4.5, '\\gamma=0.5 \\text{每步误差减半}', { size: 15, color: F.P.dim });
""")

fig("lab25-sarsa-qlearning", "cliff-path", "1.2", 320, "悬崖行走：两种理性",
    "Q-Learning 学最优路径（贴崖，回报 $\\approx -13$）但探索噪声下会坠落；SARSA 学当前策略的值（绕行，$\\approx -20$）——on-policy 的保守。",
    "贴崖最优 vs 绕行安全", """
const F = Fig(document.getElementById('fig'));
const n = 12, x0 = 40, y0 = 60, cw = 52, ch = 52;
for (let j = 0; j < n; j++) F.box({ x: x0 + j * cw, y: y0 + 2 * ch, w: cw - 5, h: ch - 5, label: '', fill: 'rgba(58,74,168,0.05)', width: 0.7 });
for (let j = 1; j < n - 1; j++) F.box({ x: x0 + j * cw, y: y0 + ch, w: cw - 5, h: ch - 5, label: '', fill: 'rgba(200,60,60,0.3)', width: 0.7 });
F.box({ x: x0, y: y0, w: cw - 5, h: ch - 5, label: 'S', fill: 'rgba(58,74,168,0.2)' });
F.box({ x: x0 + (n - 1) * cw, y: y0, w: cw - 5, h: ch - 5, label: 'G', fill: 'rgba(26,127,55,0.25)' });
// Q 路径: 贴崖
for (let j = 0; j < n - 1; j++) F.text(x0 + j * cw + 24, y0 + 1.6 * ch, '·', { size: 24, color: F.P.ink });
F.mtex(300, 45, 'Q-Learning \\text{：贴崖（} -13 \\text{，探索时坠落）}', { size: 15, color: F.P.ink, anchor: 'middle' });
// SARSA 路径: 绕一层
for (let j = 0; j < n - 1; j++) F.text(x0 + j * cw + 24, y0 + 2.6 * ch, '·', { size: 24, color: F.P.green });
F.text(300, 235, 'SARSA：绕行（−20，探索下稳）', { size: 15, color: F.P.green, anchor: 'middle' });
F.text(x0 + 5.5 * cw, y0 + ch + 26, '悬崖 −100', { size: 14, color: 'rgba(150,40,40,0.9)', anchor: 'middle' });
""")

fig("lab25-sarsa-qlearning", "return-curves", "1.2", 300, "训练期回报与执行回报",
    "Q-Learning 训练期回报更低（探索时掉崖）但最终策略更优；SARSA 反之——学什么（最优 vs 当前策略的值）与学得稳不稳是两回事。",
    "on/off-policy 的回报分化", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: -110, y1: -5, labels: ['episode', '回报'] });
F.curve(t => -95 + 82 / (1 + Math.exp(-0.08 * (t - 40))), { t0: 0, t1: 100, color: F.P.ink, width: 2.2, label: 'Q（训练期）', labelAt: 15 });
F.curve(t => -80 + 60 / (1 + Math.exp(-0.08 * (t - 30))), { t0: 0, t1: 100, color: F.P.green, width: 2.1, label: 'SARSA', labelAt: 15 });
F.mtex(45, -30, 'Q \\text{终态} -13 > SARSA -20\\n \\text{（策略本身} Q \\text{更优）}', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab26-dqn", "replay-target", "1.2", 300, "两个解药各治一病",
    "Replay 打破样本时间相关（i.i.d. 前提）、提高复用；Target Network 冻结自举目标——$y$ 不再追逐自己，复合误差被截断。",
    "解药与病灶的对应", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 40, y: 60, w: 130, h: 52, label: '样本相关', fill: 'rgba(200,60,60,0.08)' });
F.box({ x: 40, y: 150, w: 130, h: 52, label: '自举目标移动', fill: 'rgba(200,60,60,0.08)' });
F.box({ x: 320, y: 60, w: 150, h: 52, label: 'Replay Buffer', fill: 'rgba(58,74,168,0.08)' });
F.box({ x: 320, y: 150, w: 150, h: 52, label: 'Target Network θ⁻', fill: 'rgba(58,74,168,0.08)' });
F.arrow(170, 86, 320, 86, { label: '打散相关性', width: 2, labelDy: -12 });
F.arrow(170, 176, 320, 176, { label: '冻结 y', width: 2, labelDy: -12 });
F.mtex(360, 240, '\\text{致命三要素：函数近似} + \\text{自举} + off-policy', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab26-dqn", "component-ablation", "1.3", 310, "组件的阶梯",
    "裸网络剧烈震荡甚至坍塌；+replay 稳一半；+target 稳定收敛到 300+——E1 消融的预期形态（CartPole 满分 500）。",
    "逐步加组件的回报曲线", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 200, y0: 0, y1: 500, labels: ['step / 10', '回报'] });
F.curve(t => 40 + 30 * Math.sin(t * 0.8) * Math.exp(-0.004 * t) + 20 * (F.rnd() - 0.5) + 100 * Math.exp(-0.06 * t), { t0: 0, t1: 200, color: 'rgba(150,60,60,0.8)', width: 1.6, label: '裸网络', labelAt: 60 });
F.curve(t => 30 + 180 / (1 + Math.exp(-0.05 * (t - 70))) + 25 * Math.sin(t * 0.5), { t0: 0, t1: 200, color: F.P.dim, width: 1.8, label: '+replay', labelAt: 40 });
F.curve(t => 20 + 460 / (1 + Math.exp(-0.07 * (t - 60))), { t0: 0, t1: 200, color: F.P.ink, width: 2.3, label: '全量', labelAt: 110 });
""")

fig("lab27-dqn-ablation", "q-overestimation", "1.1", 320, "max 的正偏差",
    "$\\max$ 是凸算子：$\\mathbb E[\\max_i \\epsilon_i] > 0$ 且随动作数与噪声增大——Q-Learning 的 Q 值系统性高于真实回报，自举逐轮放大。",
    "E[max ε] 随动作数 k", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 16, y0: 0, y1: 2.2, labels: ['动作数 k', 'E[max ε]（σ=1）'] });
F.curve(k => Math.sqrt(2 * Math.log(k)), { t0: 1, t1: 16, color: F.P.ink, width: 2.3, label: '≈ √(2 ln k)', labelAt: 5 });
for (let k = 1; k <= 16; k++) F.vbar(k, Math.sqrt(2 * Math.log(Math.max(k, 1.0001))), { w: 0.25, fill: 'rgba(58,74,168,0.22)' });
F.mtex(6.5, 1.0, '4 \\text{动作时} \\approx 1.03\\sigma', { size: 15, color: F.P.dim });
""")

fig("lab27-dqn-ablation", "q-vs-return", "1.2", 310, "高估的可视证据",
    "DQN 的 $\\bar Q$ 系统性飘在真实回报上方；Double 解耦选/评后两者贴合——修的是偏差不是方差。",
    "Q 估计 vs 实际回报", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 60, y0: 0, y1: 500, labels: ['训练', '值'] });
F.curve(t => 200 + 5.5 * t, { t0: 4, t1: 60, color: F.P.ink, width: 2.2, label: 'DQN 的 Q̄（虚高）', labelAt: 8 });
F.curve(t => 150 + 4.8 * t, { t0: 4, t1: 60, color: F.P.green, width: 2.2, label: 'Double 的 Q̄', labelAt: 12 });
F.curve(t => 150 + 4.8 * t - 40, { t0: 4, t1: 60, color: F.P.dim, width: 1.5, label: '实际回报（参照）', labelAt: 30 });
""")

fig("lab28-reinforce", "baseline-variance", "1.2", 320, "baseline：无偏而降方差",
    "$\\mathbb E[b\\nabla\\log\\pi] = 0$（与动作无关的 $b$ 不动期望）；$G_t - b(s)$ 去中心化后，恒正回报环境里梯度不再一律推高所有动作。",
    "梯度估计的方差对比", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 4, y0: 0, y1: 10, labels: ['', 'log Var（估计量）'] });
F.vbar(0.8, 9.2, { w: 0.55, fill: 'rgba(150,60,60,0.25)' });
F.vbar(2.0, 3.1, { w: 0.55, fill: 'rgba(58,74,168,0.25)' });
F.vbar(3.2, 1.2, { w: 0.55, fill: 'rgba(26,127,55,0.25)' });
F.text(0.8, 9.8, '无 b', { size: 17, color: F.P.deep, anchor: 'middle' });
F.text(2.0, 3.7, '常数 b', { size: 17, color: F.P.ink, anchor: 'middle' });
F.mtex(3.2, 1.8, 'V(s) \\text{基线}', { size: 17, color: F.P.green, anchor: 'middle' });
F.text(2, 0.6, '期望不变、方差逐级下降', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab28-reinforce", "entropy-to-deterministic", "1.3", 300, "策略从随机到确定",
    "训练中策略熵从 $\\ln 4 \\approx 1.39$ 收缩到近 0：从均匀探索收敛到贪心——熵正则（Lab 30/31）用来给这一过程踩刹车。",
    "策略熵的训练轨迹", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: 0, y1: 1.5, labels: ['episode', '熵 (nats)'] });
F.curve(t => 0.25 + 1.1 * Math.exp(-0.06 * t), { t0: 0, t1: 100, color: F.P.ink, width: 2.3, label: 'H(π)', labelAt: 30 });
F.curve(t => Math.log(4), { t0: 0, t1: 100, color: F.P.dim, width: 1.3, rough: false, label: 'ln 4（均匀）', labelAt: 60 });
F.mtex(50, 0.5, '\\text{熵} < 0.3 \\text{：接近确定策略}', { size: 15, color: F.P.dim, anchor: 'middle' });
""")

fig("lab29-actor-critic", "ac-architecture", "1.1", 300, "Actor-Critic 双头",
    "共享骨干、两个头：策略头输出 logits、价值头输出标量 $V(s)$——优势 $\\delta = r + \\gamma V(s') - V(s)$ 同时喂两头。",
    "共享 trunk 的双头结构", """
const F = Fig(document.getElementById('fig'));
F.box({ x: 40, y: 126, w: 90, h: 48, label: 's' });
F.box({ x: 190, y: 126, w: 110, h: 48, label: '共享骨干' });
F.box({ x: 400, y: 60, w: 150, h: 48, label: '策略头 π(a|s)', fill: 'rgba(58,74,168,0.08)', size: 15 });
F.box({ x: 400, y: 192, w: 150, h: 48, label: '价值头 V(s)', fill: 'rgba(26,127,55,0.08)' });
F.arrow(130, 150, 190, 150, { width: 2 });
F.path('M 300 150 C 350 150, 350 84, 400 84', { arrow: true, color: F.P.ink, width: 1.8 });
F.path('M 300 150 C 350 150, 350 216, 400 216', { arrow: true, color: F.P.green, width: 1.8 });
F.mtex(590, 90, '\\to logits', { size: 15, color: F.P.ink, anchor: 'start' });
F.mtex(590, 222, '\\to \\text{标量}', { size: 15, color: F.P.green, anchor: 'start' });
""")

fig("lab29-actor-critic", "nstep-spectrum", "1.3", 310, "n-step：偏差-方差光谱",
    "$\\hat A^{(n)}$ 是 TD 误差的加权和：$n=1$ 低方差高偏差（critic 主导），$n \\to T$ 匹配 MC——$\\lambda$ 是这条光谱的连续插值（GAE）。",
    "n 与稳定性", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 20, y0: 0, y1: 1.15, labels: ['n-step', '训练稳定性（示意）'] });
for (const [n, col] of [[1, F.P.ink], [5, F.P.amber], [20, F.P.dim]]) {
  F.curve(t => (n === 1 ? 1.0 - 0.05 * t / 20 : n === 5 ? 1.04 - 0.12 * Math.abs(t - 6) / 14 : 1.08 - 0.3 * t / 20),
           { t0: 1, t1: 20, color: col, width: 2.0, label: 'n=' + n, labelAt: 13 });
}
F.mtex(4, 0.55, 'n \\text{大} \\to MC \\text{化} \\to \\text{震荡}', { size: 15, color: F.P.dim });
""")

fig("lab30-ppo", "clip-curve", "1.2", 330, "Clipped 目标的分段语义",
    "$J = \\min(\\rho A, \\mathrm{clip}(\\rho, 1 \\pm \\epsilon) A)$：$A > 0$ 时上限 $1+\\epsilon$（好动作别提过头）、$A < 0$ 时下限 $1-\\epsilon$——越界即梯度归零。",
    "A>0 与 A<0 两支的包络", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0.4, x1: 1.6, y0: -1.3, y1: 1.3, labels: ['ρ', 'J'] });
F.curve(t => Math.min(t, 1.2) * 0.9, { t0: 0.4, t1: 1.6, color: F.P.ink, width: 2.3, label: 'A > 0', labelAt: 0.52 });
F.curve(t => Math.max(t, 0.8) * -0.9, { t0: 0.4, t1: 1.6, color: F.P.green, width: 2.3, label: 'A < 0', labelAt: 0.52 });
F.curve(t => 0, { t0: 0.4, t1: 1.6, color: 'rgba(0,0,0,0.25)', width: 0.8, rough: false });
F.vbar(0.8, 1.28, { w: 0.006, fill: 'rgba(0,0,0,0.3)', color: 'rgba(0,0,0,0)', width: 0 });
F.vbar(1.2, 1.28, { w: 0.006, fill: 'rgba(0,0,0,0.3)', color: 'rgba(0,0,0,0)', width: 0 });
F.mtex(0.66, -1.15, '1-\\epsilon', { size: 15, color: F.P.dim, anchor: 'middle' });
F.mtex(1.34, -1.15, '1+\\epsilon', { size: 15, color: F.P.dim, anchor: 'middle' });
F.mtex(1.42, 0.7, '\\rho > 1+\\epsilon \\text{：} \\nA>0 \\text{梯度归零}', { size: 14, color: F.P.dim, anchor: 'start' });
""")

fig("lab30-ppo", "kl-earlystop", "1.4", 300, "信任域的守门员",
    "$\\mathrm{KL}(\\pi_\\theta \\| \\pi_{old})$ 超过 $1.5 \\times$ 目标值即跳过该批：epoch 复用越多越省样本，但越界即破坏信任域——回报回退的代价更大。",
    "KL 随 epoch 的膨胀与熔断", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 10, y0: 0, y1: 3.2, labels: ['epoch（批内）', 'KL'] });
F.curve(t => 0.25 * Math.exp(0.35 * t), { t0: 0, t1: 10, color: F.P.ink, width: 2.3, label: 'KL 增长', labelAt: 1.2 });
F.curve(t => 1.5, { t0: 0, t1: 10, color: 'rgba(200,60,60,0.8)', width: 1.6, rough: false, label: '熔断阈 1.5·KL_target', labelAt: 3.2 });
F.vbar(5.1, 3.15, { w: 0.05, fill: 'rgba(200,60,60,0.5)', color: 'rgba(0,0,0,0)', width: 0 });
F.text(5.1, 0.4, '早停', { size: 15, color: 'rgba(200,60,60,0.9)', anchor: 'middle' });
""")

fig("lab31-continuous-control", "sac-entropy", "1.3", 310, "最大熵：付钱买探索",
    "$J = \\mathbb E[\\sum r + \\alpha \\mathcal H]$：$\\alpha$ 大策略保持宽分布（相空间覆盖广）、回报略降；确定性策略零熵——探索外包给噪声进程。",
    "熵-回报的权衡（α 扫描）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 0.6, y0: 0, y1: 1.05, labels: ['α（熵温度）', '归一值'] });
F.curve(t => 1 - 0.4 * t, { t0: 0.02, t1: 0.58, color: F.P.ink, width: 2.3, label: '回报', labelAt: 0.42 });
F.curve(t => 0.15 + 1.5 * t, { t0: 0.02, t1: 0.58, color: F.P.amber, width: 2.1, label: '策略熵', labelAt: 0.1 });
F.text(0.3, 0.75, '能量花在探索上', { size: 16, color: F.P.dim, anchor: 'middle' });
""")

fig("lab31-continuous-control", "algo-evolution", "1.1", 310, "DDPG → TD3 → SAC 的阶梯",
    "DDPG（噪声探索 + 单 Q，失败 seed 常见）→ TD3（双 Q 取 min + 延迟更新）→ SAC（熵内建 + 随机策略）：多 seed 回归带一路收窄。",
    "三算法的多 seed 稳定性", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0, x1: 100, y0: -200, y1: -20, labels: ['step', 'Pendulum 回报'] });
F.curve(t => -170 + 60 / (1 + Math.exp(-0.07 * (t - 55))) + 40 * Math.sin(t * 0.4) * Math.exp(-0.01 * t), { t0: 0, t1: 100, color: 'rgba(150,60,60,0.7)', width: 1.7, label: 'DDPG', labelAt: 12 });
F.curve(t => -160 + 80 / (1 + Math.exp(-0.08 * (t - 40))) + 18 * Math.sin(t * 0.3), { t0: 0, t1: 100, color: F.P.dim, width: 1.9, label: 'TD3', labelAt: 8 });
F.curve(t => -150 + 120 / (1 + Math.exp(-0.09 * (t - 30))), { t0: 0, t1: 100, color: F.P.ink, width: 2.3, label: 'SAC', labelAt: 62 });
""")
