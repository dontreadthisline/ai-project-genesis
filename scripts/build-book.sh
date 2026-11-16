#!/usr/bin/env bash
# 构建书籍：quarto render + MathJax 本地化（离线/无代理环境可用）。
# 背景：Quarto 1.8 强制 MathJax v3 且从 CDN 加载，断网或 CDN 不可达时公式全部失效。
# 处理：用仓库自带的 assets/mathjax/tex-svg-full.js（SVG 输出，无字体文件依赖）替换 CDN 引用，
#       并移除 ES6 polyfill（现代浏览器不需要）。
set -euo pipefail
cd "$(dirname "$0")/.."

quarto render

mkdir -p _book/site_libs/mathjax
cp assets/mathjax/tex-svg-full.js _book/site_libs/mathjax/

python3 - <<'PY'
from pathlib import Path

book = Path("_book")
cdn_mathjax = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"

for html in book.rglob("*.html"):
    depth = len(html.relative_to(book).parts) - 1  # 相对 _book 根的目录深度
    prefix = "../" * depth
    text = html.read_text(encoding="utf-8")
    text = text.replace(cdn_mathjax, f"{prefix}site_libs/mathjax/tex-svg-full.js")
    # 移除 ES6 polyfill（cdnjs 外链）
    lines = [ln for ln in text.splitlines() if "cdnjs.cloudflare.com/polyfill" not in ln]
    html.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# 校验：不允许残留任何外部 CDN 引用
if grep -rE 'cdn\.jsdelivr\.net|cdnjs\.cloudflare' _book --include='*.html' | grep -q .; then
    echo "ERROR: 仍有 CDN 引用残留" >&2
    exit 1
fi

echo "OK: 渲染完成，MathJax 已本地化（无外部依赖）"
