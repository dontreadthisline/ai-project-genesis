#!/usr/bin/env python3
"""讲义结构统一（照 Lab 01 结构）：删 §3-§7、§8 收编为 §3、源码链接 GitHub 化。

用法: python3 tools/refactor-labs.py [--dry]
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GH = "https://github.com/dontreadthisline/ai-project-genesis/blob/main"


def split_sections(text):
    secs = {}
    for p in re.split(r"\n(?=## )", text):
        m = re.match(r"## (\d+)\. ", p)
        if m:
            secs[int(m.group(1))] = p
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    for md in sorted(ROOT.glob("docs/stage*/lab*.md")):
        text = md.read_text(encoding="utf-8")
        orig = text
        secs = split_sections(text)

        # 1) 删 §3-§7（对照实验/预期图表/观测指标/结果分析/验收清单）
        for n in (3, 4, 5, 6, 7):
            if n in secs:
                text = text.replace(secs[n] + "\n", "")
                text = text.replace(secs[n], "")

        # 2) §8 -> §3（保留原标题文本，仅改编号）
        if 8 in secs:
            text = text.replace(secs[8], secs[8].replace("## 8.", "## 3.", 1), 1)

        # 3) 源码链接 GitHub 化（相对路径 -> 绝对 URL；本地 cd 命令保留）
        text = re.sub(r"\]\(\.\./\.\./code/", f"]({GH}/code/", text)

        # 4) §0 末尾补路线行（照 Lab 01 的导览，自动版：小节标题链）
        if "## 0" in secs and "正文路线" not in text:
            titles = re.findall(r"^### (\d+\.\d+) (.+)$", text, re.M)
            if titles:
                chain = " → ".join(f"§{n} {t}" for n, t in titles)
                route = f"\n> 本讲路线：{chain}；实现与源码见 §2，跨讲锚点见 §3。\n"
                sec0 = split_sections(text).get(0)
                if sec0:
                    text = text.replace(sec0, sec0.rstrip() + "\n" + route)

        # 5) 清多余空行
        text = re.sub(r"\n{4,}", "\n\n", text)

        if text != orig:
            if not args.dry:
                md.write_text(text, encoding="utf-8")
            print(("DRY " if args.dry else "") + f"{md.parent.name}/{md.name}")


if __name__ == "__main__":
    main()
