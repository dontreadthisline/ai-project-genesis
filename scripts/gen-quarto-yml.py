#!/usr/bin/env python3
"""从 docs/manifest.yml 生成 _quarto.yml。

内容与渲染分离的约定：
  - docs/manifest.yml 是书籍结构的单一事实源；
  - 本脚本把该结构翻译为 Quarto 的配置（_quarto.yml 为生成物，不入库）；
  - 切换其他构建工具时，为本工具新写一个生成器（如 gen-vitepress.py），
    内容文件与 manifest 不需要任何改动。
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "manifest.yml"

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

# Quarto book 硬性要求 home page 必须是 index.*：
# 从 manifest 的 landing 复制出 index.qmd（每次构建重新生成，属渲染层产物）
import shutil

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
        "html": {
            "theme": ["litera", "assets/quarto-theme.scss"],
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
print(f"generated {out} ({len(parts)} parts)")
