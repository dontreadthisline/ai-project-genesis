#!/usr/bin/env python3
"""从 docs/manifest.yml 生成 VitePress 站点配置（site-vitepress/）。

与 gen-quarto-yml.py 平行的渲染层生成器：证明内容/渲染分离的承诺——
切换构建工具时，docs/ 与 code/ 零改动，只新增一个生成器。

用法：
    python3 scripts/gen-vitepress.py          # 生成 site-vitepress/.vitepress/config.mts 等
    cd site-vitepress && npm install && npm run build
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site-vitepress"
MANIFEST = yaml.safe_load((ROOT / "docs" / "manifest.yml").read_text(encoding="utf-8"))

book = MANIFEST["book"]
parts = MANIFEST.get("parts", [])

# ---------------------------------------------------------------- sidebar 构造


def link_for(chapter: str) -> str:
    """docs/a/b.md -> /a/b（cleanUrls，landing 由 rewrites 处理）"""
    rel = chapter.removeprefix("docs/").removesuffix(".md")
    return "/" + rel


sidebar = []
for part in parts:
    items = [{"text": p.stem, "link": link_for(str(p))} for p in map(Path, part["chapters"])]
    # 组内文本：README 显示为「导览」
    for it in items:
        if it["text"] == "README":
            it["text"] = "本阶段导览"
    sidebar.append({"text": part["title"], "collapsed": False, "items": items})

sidebar_js = json.dumps(sidebar, ensure_ascii=False, indent=2)

config_mts = f"""// 由 scripts/gen-vitepress.py 从 docs/manifest.yml 生成（勿手改）
import {{ defineConfig }} from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({{
  title: {json.dumps(book['title'], ensure_ascii=False)},
  description: {json.dumps(book['subtitle'], ensure_ascii=False)},
  srcDir: '../docs',
  outDir: 'dist',
  cleanUrls: true,
  rewrites: {{ 'preface.md': 'index.md' }},
  markdown: {{
    config(md) {{ md.use(mathjax3) }},
  }},
  head: [['link', {{ rel: 'stylesheet', href: '/vp.css' }}]],
  themeConfig: {{
    outline: [2, 3],
    sidebar: {sidebar_js},
  }},
}})
"""

package_json = json.dumps(
    {
        "name": "genesis-site-vitepress",
        "private": True,
        "type": "module",
        "scripts": {"build": "vitepress build", "dev": "vitepress dev"},
        "devDependencies": {
            "vitepress": "^1.6.3",
            "markdown-it-mathjax3": "^4.3.2",
        },
    },
    ensure_ascii=False,
    indent=2,
)

vp_css = """/* 中文长文阅读微调（叠加在 VitePress 默认主题上） */
:root {
  --vp-font-family-base: -apple-system, "PingFang SC", "Hiragino Sans GB",
    "Noto Sans SC", "Microsoft YaHei", sans-serif;
}
.VPContent main { max-width: 52rem; margin: 0 auto; }
.vp-doc { line-height: 1.85; font-size: 16px; }
.vp-doc h1, .vp-doc h2 { letter-spacing: 0.01em; }
.vp-doc p { margin: 1.05em 0; }
.vp-doc table { display: table; width: auto; min-width: 60%; }
.vp-doc .MathJax_Display, mjx-container[display="true"] {
  overflow-x: auto; overflow-y: hidden; padding-bottom: 2px; margin: 1.1em 0;
}
"""

vitepress_dir = SITE / ".vitepress"
vitepress_dir.mkdir(parents=True, exist_ok=True)
(vitepress_dir / "config.mts").write_text(config_mts, encoding="utf-8")
(SITE / "package.json").write_text(package_json + "\n", encoding="utf-8")
(SITE / "vp.css").write_text(vp_css, encoding="utf-8")
(SITE / ".gitignore").write_text("node_modules/\ndist/\n.cache/\n", encoding="utf-8")

# VitePress 需要 public/ 目录（即使为空）
(SITE / "public").mkdir(exist_ok=True)
print(f"generated {vitepress_dir / 'config.mts'} ({len(parts)} parts) + package.json")
