#!/usr/bin/env python3
"""从 docs/manifest.yml 生成 _quarto.yml，并安装主题扩展。

内容与渲染分离的约定：
  - docs/manifest.yml 是书籍结构的单一事实源；
  - 主题是标准 Quarto 扩展（render/themes/<name>/），本项目安装到 _extensions/ 使用，
    外部 Quarto 项目可直接 `quarto add <发布的主题仓库>` 使用同一份主题；
  - 换主题：环境变量 GENESIS_THEME=<name>（默认 genesis-book）。
"""
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "docs" / "manifest.yml"

THEME = os.environ.get("GENESIS_THEME", "genesis-book")
THEME_DIR = ROOT / "render" / "themes" / THEME
EXT_DIR = ROOT / "_extensions" / THEME

manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
book = manifest["book"]
landing = manifest["landing"]
parts = manifest.get("parts", [])

# 校验：manifest 引用的每个章节文件必须存在
missing = [
    ch
    for part in parts
    for ch in part["chapters"]
    if not (ROOT / ch).is_file()
]
if not (ROOT / landing).is_file():
    missing.append(landing)
if missing:
    print("ERROR: manifest 引用了不存在的文件:", file=sys.stderr)
    for m in missing:
        print(f"  - {m}", file=sys.stderr)
    sys.exit(1)

# 校验并安装主题扩展（Quarto 约定：项目 _extensions/ 目录）
if not (THEME_DIR / "_extension.yml").is_file():
    print(f"ERROR: 主题扩展不完整: render/themes/{THEME}（缺 _extension.yml）", file=sys.stderr)
    sys.exit(1)
if EXT_DIR.exists():
    shutil.rmtree(EXT_DIR)
shutil.copytree(THEME_DIR, EXT_DIR)

# Quarto book 硬性要求 home page 必须是 index.*：
# 从 manifest 的 landing 复制出 index.qmd（每次构建重新生成，属渲染层产物）
shutil.copyfile(ROOT / landing, ROOT / "index.qmd")

chapters = ["index.qmd"]
for part in parts:
    chapters.append({"part": part["title"], "chapters": part["chapters"]})

quarto = {
    "project": {"type": "book", "output-dir": "_book"},
    "book": {
        "title": book["title"],
        "subtitle": book["subtitle"],
        "search": True,
        "chapters": chapters,
    },
    "format": {
        # format 名由主题扩展的 _extension.yml 提供：<id>-html（Quarto 约定：
        # 剥离 -html 后缀定位 _extensions/<id>/，样式相关默认值全在扩展内），
        # 此处只补充项目级选项
        f"{THEME}-html": {
            "lang": "zh",
            "toc": True,
            "toc-depth": 3,
            "number-sections": False,
        }
    },
}

out = ROOT / "_quarto.yml"
out.write_text(
    yaml.safe_dump(quarto, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
print(f"generated {out} ({len(parts)} parts, theme={THEME})")
