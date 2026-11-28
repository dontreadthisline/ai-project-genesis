#!/usr/bin/env python3
"""讲义配图生成器：每章定义 2-3 张图（基于 _lib.js 图库），生成 html 图页并插入讲义。

用法: python3 tools/gen-figs.py [lab前缀过滤...]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "resources" / "figures"
DOCS = ROOT / "docs"

TPL = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @font-face {{ font-family: 'Caveat'; font-style: normal; font-weight: 500;
    src: url(../fonts/caveat-500.woff2) format('woff2'); }}
  body {{ margin: 0; font: 15px/1.6 -apple-system, "PingFang SC", sans-serif; color: #2c3a8c; }}
  .fig {{ display: block; margin: 0 auto; }}
  .figwrap {{ position: relative; width: 720px; margin: 0 auto; }}
  .mtx {{ font-family: -apple-system, "PingFang SC", sans-serif; }}
  .caption {{ text-align: center; padding: 4px 24px 10px; color: #5a6078; }}
</style>
<script src="../js/sketch.js"></script>
<script src="_lib.js"></script>
<script>
  window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }},
                     svg: {{ fontCache: 'global' }} }};
</script>
<script defer src="../../site_libs/mathjax/tex-svg-full.js"></script>
</head>
<body>
<div class="figwrap">
<svg id="fig" class="fig" width="720" height="{height}" viewBox="0 0 720 {height}"></svg>
</div>
<div class="caption">{caption}</div>
<script>
{js}
</script>
</body>
</html>
"""

IFRAME = ('<iframe src="resources/figures/{name}.html" '
          'style="width:100%;height:{iheight}px;border:1px solid rgba(58,74,168,.18);border-radius:8px" '
          'title="{title}"></iframe>\n\n<p align="center"><b>{figlabel}</b>：{capshort}</p>\n')

# 每章图定义: name / after(小节号, iframe 插到该小节末) / height / title / caption / js
FIGS = {}


def fig(lab, name, after, height, title, caption, capshort, js):
    FIGS.setdefault(lab, []).append(
        dict(name=name, after=after, height=height, iheight=height + 64,
             title=title, caption=caption, capshort=capshort, js=js))


# ============================== stage 1 ==============================

fig("lab02-joint-and-conditional", "conditional-slice", "1.2", 340, "条件分布 = 联合的竖切",
    "固定 $x = x_0$ 的窄带内的 $y$ 分布即 $p(y \\mid x)$：均值随 $\\rho x_0$ 平移、方差收窄为 $1 - \\rho^2$（$\\rho = 0.8$）。",
    "联合云的竖切给出条件分布（ρ=0.8）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3.2, x1: 3.2, y0: -2.6, y1: 2.6, labels: ['x', 'y'] });
F.cloud(160, z => [z[0] * 0.95, z[1] * 0.95], { color: F.P.dim, r: 2 });
F.ellipse({ rx: 1.9, ry: 1.0, rot: 0.68, opacity: 0.5 });
// x=1.2 窄带: 半透明竖带 + 带内样本重画深色
F.vbar(1.2, 2.55, { w: 0.7, fill: 'rgba(191,135,0,0.14)', color: 'rgba(0,0,0,0)', width: 0 });
for (let i = 0; i < 600; i++) {
  const z1 = F.gauss(), z2 = F.gauss();
  const x = z1 * 0.95, y = 0.8 * x + 0.57 * z2;   // rho=0.8
  if (Math.abs(x - 1.2) < 0.35) F.text(x, y, '·', { size: 19, color: F.P.amber });
}
F.mtex(1.35, 2.3, 'x = x_0 \\text{窄带}', { size: 17, color: F.P.amber });
F.mtex(2.7, 1.55, 'p(y | x_0)', { size: 16, color: F.P.deep, anchor: 'start' });
// 右缘内嵌条件密度(钟形点列, 均值0.96 方差0.36)
for (let i = 0; i <= 26; i++) {
  const t = -1.9 + 3.4 * i / 26;
  const v = Math.exp(-((t - 0.96) ** 2) / (2 * 0.36));
  F.text(2.62 + v * 0.42, t, '·', { size: 19, color: F.P.deep });
}
""")

fig("lab02-joint-and-conditional", "same-marginals", "1.3", 320, "相同边缘、不同联合",
    "两个分布的边缘都是 $\\mathcal N(0,1)$，联合却是 $+$ 形与 $-$ 形——边缘对联合的信息为零（讲义反例 B）。",
    "X 形混合: 边缘相同、联合相反", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3, x1: 3, y0: -3, y1: 3, labels: ['x', 'y'] });
for (const sg of [1, -1]) {
  // 两支: 均值 0, 协方差 [[1, .9*sg],[.9*sg, 1]] 各 90 点
  const rot = sg > 0 ? Math.PI / 4 : -Math.PI / 4;
  F.cloud(80, z => {
    const x = z[0] * 0.9 + z[1] * 0.44, y = (z[0] * 0.9 + z[1] * 0.44) * sg; // 近似斜切
    return [x, y];
  }, { color: sg > 0 ? F.P.ink : F.P.green, r: 2.1 });
  F.ellipse({ rx: 1.85, ry: 0.75, rot, color: sg > 0 ? F.P.ink : F.P.green, opacity: 0.6 });
}
F.mtex(0, -2.5, '\\text{整体协方差} \\approx 0 \\text{，但结构全在相关性里}', { size: 17, color: F.P.dim, anchor: 'middle' });
""")

fig("lab03-monte-carlo", "convergence", "1.2", 330, "Monte Carlo 收敛速率",
    "估计误差随 $n$ 按 $1/\\sqrt{n}$ 衰减（对数-对数视角斜率 $-1/2$）：精度提高一个量级，样本要增加两个量级。",
    "MC 误差 ∝ 1/√n（log-log）", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 1, x1: 6, y0: -2.5, y1: -0.4, labels: ['log n', 'log 误差'] });
F.curve(n => -0.5 * n - 0.35, { t0: 1.3, t1: 5.8, color: F.P.amber, width: 2.4, rough: false,
                                 label: '斜率 −1/2（1/√n）', labelAt: 1.7 });
// 模拟重复实验的误差点: 散布在理论线附近
for (let i = 0; i < 24; i++) {
  const n = 1.4 + F.rnd() * 4.2;
  const err = -0.5 * n - 0.35 + (F.rnd() - 0.5) * 0.5;
  F.text(n, err, '·', { size: 20, color: F.P.ink });
}
F.text(3.6, -2.15, '每次重复实验一个点', { size: 16, color: F.P.dim });
""")

fig("lab03-monte-carlo", "clt", "1.4", 330, "估计量自身的分布（CLT）",
    "重复实验 $10^3$ 次的 $\\hat\\mu_n$ 直方图近似正态——无论 $f$ 的原始分布多么偏斜（CLT 的威力）。",
    "估计量的抽样分布 → 正态", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -3.4, x1: 3.4, y0: 0, y1: 0.5, labels: ['(μ̂ − μ)/se', '频率'] });
// 正态直方 bins（预计算 24 bins 的正态形状）
const bins = 24, h = 6.4 / bins, n = 1000;
const counts = new Array(bins).fill(0);
for (let i = 0; i < n; i++) {
  const t = F.gauss();
  const k = Math.floor((t + 3.2) / h);
  if (k >= 0 && k < bins) counts[k]++;
}
for (let k = 0; k < bins; k++) F.vbar(-3.2 + (k + 0.5) * h, counts[k] / (n * h));
F.curve(x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI),
        { t0: -3.2, t1: 3.2, color: F.P.amber, width: 2.4, rough: false, label: 'N(0,1)' });
""")

fig("lab04-entropy-kl", "h-ce-kl", "1.1", 330, "交叉熵 = 熵 + KL",
    "$P = (0.7, 0.3)$ 固定，$Q = (t, 1-t)$ 扫描：$H(P, Q) = H(P) + D_{KL}$ 的最小值恰在 $t = 0.7$，落在水平线 $H(P) \\approx 0.611$ 上。",
    "三曲线: 最小值在 Q=P 处", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 0.02, x1: 0.98, y0: 0, y1: 1.4, labels: ['t', 'nats'] });
const H = (a, b) => -(a * Math.log(a) + b * Math.log(b));
F.curve(t => H(0.7, 0.3), { t0: 0.02, t1: 0.98, color: F.P.green, width: 1.8, rough: false,
                            label: 'H(P)（水平线）', labelAt: 0.2 });
F.curve(t => H(0.7, 0.3) + (0.7 * Math.log(0.7 / t) + 0.3 * Math.log(0.3 / (1 - t))),
        { t0: 0.02, t1: 0.98, color: F.P.ink, width: 2.2, label: 'H(P,Q)', labelAt: 0.33 });
F.curve(t => 0.7 * Math.log(0.7 / t) + 0.3 * Math.log(0.3 / (1 - t)),
        { t0: 0.02, t1: 0.98, color: F.P.amber, width: 1.8, label: 'D_KL', labelAt: 0.3 });
F.vbar(0.7, 1.35, { w: 0.012, fill: 'rgba(0,0,0,0.25)', color: 'rgba(0,0,0,0)', width: 0 });
F.mtex(0.72, 0.12, 't = 0.7 = P(1)', { size: 16, color: F.P.deep, anchor: 'start' });
""")

fig("lab04-entropy-kl", "kl-asymmetry", "1.3", 300, "KL 的不对称",
    "$D_{KL}(P \\| Q) \\ne D_{KL}(Q \\| P)$：前向惩罚 $Q$ 遗漏 $P$ 的支撑（mode-covering），反向允许 $Q$ 集中（mode-seeking）——DPO 与 VAE 各取其一。",
    "双向 KL: 两个不同的量", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: -1, x1: 7, y0: 0, y1: 1.8, labels: ['', 'nats'] });
const pairs = [
  { x: 0.6, fwd: 1.614, rev: 0.637, tag: 'P=(.5,.5) Q=(.99,.01)' },
  { x: 3.4, fwd: 0.51, rev: 0.51, tag: 'P=Q（相等）' },
  { x: 5.6, fwd: 0.09, rev: 3.9, tag: 'P=(.9,.1) Q=(.5,.5)' },
];
for (const p of pairs) {
  F.vbar(p.x, p.fwd, { w: 0.5, fill: 'rgba(58,74,168,0.25)' });
  F.vbar(p.x + 0.9, p.rev, { w: 0.5, fill: 'rgba(26,127,55,0.25)' });
  F.mtex(p.x + 0.7, -0.32, p.tag.replace('（相等）', '\\text{（相等）}'), { size: 13, color: '#9aa1b9' });
}
F.mtex(0.75, 1.7, 'D(P\\|Q)', { size: 17, color: F.P.deep });
F.mtex(4.3, 1.7, 'D(Q\\|P)', { size: 17, color: F.P.green });
""")

fig("lab05-mle", "likelihood-contour", "1.2", 360, "对数似然的地形",
    "固定 30 个样本（真值 $\\mu = 3, \\sigma^2 = 4$），$\\ell(\\mu, \\sigma^2)$ 的等值线是同心椭圆族，峰值即 MLE 闭式解 $\\bar x$。",
    "logL(μ, σ²) 等值线与 MLE 峰", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 2.3, x1: 3.7, y0: 1.6, y1: 6.4, labels: ['μ', 'σ²'] });
const lvls = [{ r: 1, op: 0.75 }, { r: 2, op: 0.5 }, { r: 3.2, op: 0.32 }];
for (const l of lvls)
  F.ellipse({ cx: 3, cy: 4, rx: 0.13 * l.r, ry: 0.42 * l.r, opacity: l.op, width: 1.6 });
F.mtex(3.02, 4.1, '\\mû = x̄', { size: 17, color: F.P.deep, anchor: 'start' });
// 真值标记
F.node(0, 0, {}); // noop
F.text(2.52, 5.6, '真值 (3, 4)', { size: 15, color: F.P.dim, anchor: 'start' });
F.ellipse({ cx: 2.62, cy: 5.55, rx: 0.06, ry: 0.06, color: F.P.amber, width: 1.5 });
""")

fig("lab05-mle", "bias", "1.3", 320, "σ̂² 的偏差与修正",
    "$\\hat\\sigma^2_{1/n}$ 系统性低估（$-\\sigma^2/n$），$1/(n-1)$ 版本无偏：小样本下两者差异显著，随 $n$ 消失。",
    "1/n 有偏 vs 1/(n−1) 无偏", """
const F = Fig(document.getElementById('fig'));
F.axis({ x0: 2, x1: 60, y0: 0.5, y1: 1.05, labels: ['n', 'E[σ̂²]/σ²'] });
F.curve(n => 1 - 1 / n, { t0: 2, t1: 60, color: F.P.ink, width: 2.2, label: '1/n 版', labelAt: 8 });
F.curve(n => 1, { t0: 2, t1: 60, color: F.P.green, width: 1.8, rough: false, label: '1/(n−1) 版', labelAt: 20 });
F.mtex(6, 0.62, 'n=5 \\text{时低估} 20%', { size: 16, color: F.P.dim });
""")



# ---- 加载各阶段图定义（tools/figs-stage*.py 逐个 exec，共享上方 fig/FIGS） ----
for defs_file in sorted(Path(__file__).parent.glob("figs-stage*.py")):
    exec(compile(defs_file.read_text(encoding="utf-8"), str(defs_file), "exec"))


def gen(lab_prefixes=None):
    made, inserted = 0, 0
    for lab, figs in FIGS.items():
        if lab_prefixes and not any(lab.startswith(p) for p in lab_prefixes):
            continue
        # 找讲义
        md = next(DOCS.glob(f"stage*/{lab}.md"), None)
        if not md:
            print(f"WARN: 讲义不存在 {lab}")
            continue
        text = md.read_text(encoding="utf-8")
        # 生成图页
        for i, f in enumerate(figs, 1):
            html = TPL.format(title=f["title"], height=f["height"], caption=f["caption"], js=f["js"])
            (FIGDIR / f"{f['name']}.html").write_text(html, encoding="utf-8")
            made += 1
        # 插入 iframe: 在 after 小节的末尾（下一个 ###/## 标题前）
        for i, f in enumerate(figs, 1):
            figlabel = f"图 {lab[3:5]}.{i}"
            block = IFRAME.format(name=f["name"], iheight=f["iheight"], title=f["title"],
                                  figlabel=figlabel, capshort=f["capshort"])
            pat = re.compile(rf"(### {f['after']} .*?\n.*?)(?=\n### |\n## |\Z)", re.S)
            m = pat.search(text)
            if not m:
                print(f"WARN: {lab} 未找到小节 {f['after']}")
                continue
            if f'figures/{f["name"]}' in text:
                continue  # 已插入
            text = text[:m.end(1)] + "\n" + block + text[m.end(1):]
            inserted += 1
        md.write_text(text, encoding="utf-8")
    print(f"生成图页 {made}，插入 iframe {inserted}")


if __name__ == "__main__":
    gen(sys.argv[1:] or None)
