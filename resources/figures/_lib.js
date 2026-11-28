/* _lib.js — 讲义配图库：基于 sketch.js 的手绘图表工厂
 * 依赖: ../js/sketch.js（先于本文件加载）
 * 两套坐标: axis() 后的 F.curve/cloud/ellipse/vbar/heat 用数据坐标;
 *           F.box/arrow/text/layer 用像素坐标（架构图/流程图）。
 */
const Fig = (() => {
  // 确定性伪随机（图可复现）
  let _seed = 20260101;
  const rnd = () => (_seed = (_seed * 48271) % 2147483647) / 2147483647;
  const gauss = () => {
    const u = Math.max(rnd(), 1e-9), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  function Fig(svg) {
    const s = Sketch(svg);
    const P = s.PALETTE;
    const W = +svg.getAttribute('width') || 720, H = +svg.getAttribute('height') || 360;
    let map = null; // 数据坐标映射

    const self = {
      P, svg, rnd, gauss, W, H,

      axis({ x0 = -3, x1 = 3, y0 = -1, y1 = 1, pad = [56, 18, 40, 18], labels = ['x', 'y'], bare = false } = {}) {
        const [pt, pr, pb, pl] = pad;
        const px = t => pt + (t - x0) / (x1 - x0) * (W - pt - pr);
        const py = v => H - pb - (v - y0) / (y1 - y0) * (H - pb - pt);
        map = { px, py, x0, x1, y0, y1 };
        if (bare) return self;
        s.path(`M ${pt - 10} ${H - pb} L ${W - pr + 6} ${H - pb}`, { arrow: true, opacity: 0.65, rough: false });
        s.path(`M ${pt} ${H - pb + 8} L ${pt} ${pt - 8}`, { arrow: true, opacity: 0.65, rough: false });
        if (labels[0]) s.label(pt - 22, H - pb + 16, labels[0], { mono: true, size: 15, color: P.ink });
        if (labels[1]) s.label(pt - 10, pt - 16, labels[1], { mono: true, size: 15, color: P.ink });
        return self;
      },

      curve(f, { t0, t1, n = 70, color = P.ink, width = 2, rough = true, opacity = 1, label, labelAt } = {}) {
        const { px, py } = map;
        let d = '';
        for (let i = 0; i <= n; i++) {
          const t = t0 + (t1 - t0) * i / n;
          d += (i ? 'L' : 'M') + ` ${px(t).toFixed(1)} ${py(f(t)).toFixed(1)} `;
        }
        s.path(d, { color, width, rough, opacity });
        if (label) {
          const lt = labelAt === undefined ? t0 + 0.7 * (t1 - t0) : labelAt;
          s.label(px(lt), py(f(lt)) - 12, label, { size: 18, color });
        }
        return self;
      },

      cloud(n, transform = z => z, { color = P.dim, r = 2.2, opacity = 1 } = {}) {
        const { px, py } = map;
        for (let i = 0; i < n; i++) {
          const [x, y] = transform([gauss(), gauss()]);
          s.dot(px(x), py(y), color, r, opacity);
        }
        return self;
      },

      ellipse({ cx = 0, cy = 0, rx = 1, ry = 1, rot = 0, color = P.ink, width = 1.8, opacity = 1, fill = 'none', rough = true, label } = {}) {
        const { px, py } = map;
        let d = '';
        for (let i = 0; i <= 56; i++) {
          const a = 2 * Math.PI * i / 56;
          const x = rx * Math.cos(a), y = ry * Math.sin(a);
          const X = px(cx + x * Math.cos(rot) - y * Math.sin(rot));
          const Y = py(cy + x * Math.sin(rot) + y * Math.cos(rot));
          d += (i ? 'L' : 'M') + ` ${X.toFixed(1)} ${Y.toFixed(1)} `;
        }
        s.path(d + 'Z', { color, width, opacity, fill, rough });
        if (label) s.label(px(cx + rx * 0.75), py(cy + ry * 0.75), label, { size: 17, color });
        return self;
      },

      vbar(x, h, { w = 0.6, fill = 'rgba(58,74,168,0.16)', color = P.ink, width = 1.2, rough = true } = {}) {
        const { px, py } = map;
        const xL = px(x - w / 2), xR = px(x + w / 2), y0 = py(map.y0), y1 = py(h);
        s.path(`M ${xL} ${y0} L ${xL} ${y1} L ${xR} ${y1} L ${xR} ${y0}`, { fill, color, width, rough });
        return self;
      },

      heat(grid, { x = 0, y = 0, cw = 1, ch = 1, gap = 0.08, palette } = {}) {
        // grid: 二维数组的数（0..1）；数据坐标左上角 (x,y)，每格 cw×ch
        const { px, py } = map;
        for (let i = 0; i < grid.length; i++) {
          for (let j = 0; j < grid[i].length; j++) {
            const v = grid[i][j];
            const X = px(x + j * cw), Y = py(y + (i + 1) * ch);
            const color = palette ? palette(v) : `rgba(58,74,168,${(0.12 + 0.8 * v).toFixed(2)})`;
            s.path(`M ${X} ${Y} L ${X + px(x + (j + 1) * cw) - X - 2} ${Y} L ${X + px(x + (j + 1) * cw) - X - 2} ${Y + py(y + i * ch) - Y - 2} L ${X} ${Y + py(y + i * ch) - Y - 2} Z`,
                   { fill: color, width: 0.5, color: 'rgba(0,0,0,0.12)', rough: false });
          }
        }
        return self;
      },

      // ---- 像素坐标（架构/流程图） ----
      box({ x, y, w = 120, h = 44, label, color = P.ink, fill = 'rgba(58,74,168,0.07)', width = 1.8, size = 17, rough = true } = {}) {
        s.path(`M ${x} ${y} L ${x + w} ${y} L ${x + w} ${y + h} L ${x} ${y + h} Z`, { color, fill, width, rough });
        if (label) s.label(x + w / 2, y + h / 2 + 6, label, { size, color: P.deep, anchor: 'middle' });
        return self;
      },

      arrow(x1, y1, x2, y2, { label, color = P.amber, width = 2, labelDy = -10, flow: useFlow = false, dur = 2 } = {}) {
        const d = `M ${x1} ${y1} L ${x2} ${y2}`;
        s.path(d, { color, width, arrow: true });
        if (useFlow) s.flow(d, { color, dur });
        if (label) s.label((x1 + x2) / 2, (y1 + y2) / 2 + labelDy, label, { size: 17, color });
        return self;
      },

      text(x, y, str, opt = {}) { s.label(x, y, str, opt); return self; },

      // 图内文字一律 LaTeX 渲染（规范：公式与文字不使用手写字体/unicode 拼接）
      mtex(x, y, latex, { size = 17, color = '#2c3a8c', anchor = 'middle' } = {}) {
        const wrap = svg.parentElement;
        if (!wrap || !wrap.classList.contains('figwrap')) return self;
        const span = document.createElement('span');
        span.className = 'mtx';
        span.style.cssText = `position:absolute; left:${x}px; top:${y}px; font-size:${size}px; color:${color}; white-space:nowrap;`;
        span.textContent = '$' + latex + '$';
        wrap.appendChild(span);
        if (anchor === 'middle') span.style.transform = 'translate(-50%, -50%)';
        else if (anchor === 'start') span.style.transform = 'translate(0, -50%)';
        else if (anchor === 'end') span.style.transform = 'translate(-100%, -50%)';
        if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([span]);
        return self;
      },

      path(d, opt = {}) { s.path(d, opt); return self; },
      dot(x, y, color, r, opacity) { s.dot(x, y, color, r, opacity); return self; },

      layer({ x, y, n, r = 5, spread = 66, label, color = P.ink } = {}) {
        const ys = Array.from({ length: n }, (_, i) => y + (i - (n - 1) / 2) * spread);
        for (const yy of ys) s.dot(x, yy, color, r);
        if (label) s.label(x, y + spread * (n - 1) / 2 + 20, label, { size: 16, color: P.dim, anchor: 'middle' });
        return ys;
      },

      links(ys1, ys2, x1, x2, { opacity = 0.25, step = 1 } = {}) {
        for (let i = 0; i < ys1.length; i += step)
          for (let j = 0; j < ys2.length; j += step)
            s.path(`M ${x1} ${ys1[i]} L ${x2} ${ys2[j]}`, { opacity, width: 0.8, color: P.dim, rough: false });
        return self;
      },

      node(x, y, { label, color = P.ink, r = 5.5 } = {}) {
        s.dot(x, y, '#fff', r);
        s.node && 0; // node 已由 sketch 提供
        s.path(`M ${x - r} ${y} A ${r} ${r} 0 1 1 ${x + r} ${y} A ${r} ${r} 0 1 1 ${x - r} ${y}`, { color, width: 1.6, rough: true, fill: '#fff' });
        if (label) s.label(x, y - r - 8, label, { size: 16, color, anchor: 'middle' });
        return self;
      },
    };
    return self;
  }
  return Fig;
})();
