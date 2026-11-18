#!/usr/bin/env bash
# 构建书籍（渲染层唯一入口）：
#   1. docs/manifest.yml -> _quarto.yml + 主题扩展安装到 _extensions/（gen-quarto-yml.py）
#   2. quarto render -> _book/
#   3. MathJax 本地化：CDN 引用替换为自托管 book_render/vendor/mathjax-tex-svg-full.js
#      （SVG 输出，无字体文件依赖），并移除 ES6 polyfill，断网/无代理环境公式仍可渲染
#   4. 校验：无外部 CDN 残留、无未重写的 .md 链接
#
# 换主题：GENESIS_THEME=paper-book ./book_render/scripts/build-book.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 book_render/scripts/gen-quarto-yml.py
quarto render

mkdir -p _book/site_libs/mathjax
cp book_render/vendor/mathjax-tex-svg-full.js _book/site_libs/mathjax/

python3 - <<'PY'
from pathlib import Path

book = Path("_book")
cdn_mathjax = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"

for html in book.rglob("*.html"):
    depth = len(html.relative_to(book).parts) - 1  # 相对 _book 根的目录深度
    prefix = "../" * depth
    text = html.read_text(encoding="utf-8")
    text = text.replace(cdn_mathjax, f"{prefix}site_libs/mathjax/tex-svg-full.js")
    lines = [ln for ln in text.splitlines() if "cdnjs.cloudflare.com/polyfill" not in ln]
    html.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# 校验 1：不允许残留任何外部 CDN 引用
if grep -rE 'cdn\.jsdelivr\.net|cdnjs\.cloudflare' _book --include='*.html' | grep -q .; then
    echo "ERROR: 仍有 CDN 引用残留" >&2
    exit 1
fi

# 校验 2：不允许残留未重写的 .md 链接（死链）
if grep -rE 'href="[^"]*\.md"' _book --include='*.html' | grep -q .; then
    echo "ERROR: 存在未重写的 .md 链接" >&2
    exit 1
fi

echo "OK: 渲染完成，MathJax 已本地化，无外部依赖，无死链"
