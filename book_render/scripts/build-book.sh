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
cp book_render/vendor/mathjax-tex-svg-full.js _book/site_libs/mathjax/tex-svg-full.js
# 项目级 resources/（如主题手绘原语的字体与脚本）复制进站点根
if [ -d resources ]; then
  rm -rf _book/resources
  cp -r resources _book/resources
fi
# 实验源码进站（讲义链接 code/... 需在站点内可解析）
if [ -d code ]; then
  rm -rf _book/code
  rsync -a --exclude '.venv' --exclude 'output' --exclude '__pycache__' \
    --exclude '.DS_Store' --exclude 'uv.lock' --exclude '.python-version' code _book/
fi

python3 - <<'PY'
from pathlib import Path

book = Path("_book")
cdn_mathjax = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"

for html in book.rglob("*.html"):
    depth = len(html.relative_to(book).parts) - 1  # 相对 _book 根的目录深度
    prefix = "../" * depth
    text = html.read_text(encoding="utf-8")
    text = text.replace(cdn_mathjax, f"{prefix}site_libs/mathjax/tex-svg-full.js")
    # 项目 resources 引用按页面深度改写（同 MathJax 的相对路径问题）
    text = text.replace('="resources/', f"=\"{prefix}resources/")
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

# 校验 3：全部本地相对引用必须真实存在（site_libs / code / resources，防静默 404）
python3 - <<'PY'
import re, sys
from pathlib import Path
book = Path("_book")
missing = set()
for html in book.rglob("*.html"):
    for m in re.finditer(r'(?:src|href)="([^"#?]+)"', html.read_text(encoding="utf-8")):
        ref = m.group(1)
        if ref.startswith(("http", "//", "mailto:", "data:")):
            continue
        if not ref.strip():
            continue
        target = (html.parent / ref).resolve()
        if not target.exists():
            missing.add(f"{html.relative_to(book)} -> {ref}")
if missing:
    print("ERROR: 本地引用 404:", file=sys.stderr)
    for m in sorted(missing):
        print(f"  {m}", file=sys.stderr)
    sys.exit(1)
PY

echo "OK: 渲染完成，MathJax 已本地化，无外部依赖，无死链，无本地 404"
