#!/usr/bin/env python3
"""从讲义生成实验代码骨架，并瘦身讲义的实现任务/观测指标两节。

用法:
    python3 tools/gen-lab-skeletons.py --only lab06    # 单个实验（试跑）
    python3 tools/gen-lab-skeletons.py                 # 全量（lab06-45）
    python3 tools/gen-lab-skeletons.py --dry           # 只打印计划不写盘

对每份讲义做三件事:
1. §2 实现任务: 接口骨架代码块 -> code/<lab>/skeleton.py;
   自测断言代码块 -> code/<lab>/selftest.py; 讲义中代码块替换为
   实现清单(提取 class/def + docstring 首行) + 源码链接。
2. 生成 code/<lab>/{pyproject.toml, README.md, run_labXX.py}。
3. §5 观测指标: JSON schema 块替换为键路径平铺清单。

stage1(lab01-05) 特判: code/ 项目已存在, 只做讲义瘦身, 链接指向既有模块文件。
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CODE = ROOT / "code"

STAGE_DEPS = {
    2: ["mlx", "numpy>=2.0", "matplotlib>=3.8"],
    3: ["mlx", "numpy>=2.0", "matplotlib>=3.8"],
    4: ["mlx", "numpy>=2.0", "matplotlib>=3.8", "gymnasium"],
    5: ["mlx", "numpy>=2.0", "matplotlib>=3.8"],
    6: ["mlx", "numpy>=2.0", "matplotlib>=3.8"],
}

LAB_RE = re.compile(r"lab(\d{2})-([a-z0-9-]+)\.md$")


def labs():
    out = []
    for stage_dir in sorted(DOCS.glob("stage*")):
        for md in sorted(stage_dir.glob("lab*.md")):
            m = LAB_RE.search(md.name)
            if m:
                out.append((int(m.group(1)), md))
    return out


def split_sections(text):
    """返回 {节号: (标题行, 内容)}，节号取自 '## N. ' 标题。"""
    secs = {}
    parts = re.split(r"\n(?=## )", text)
    for p in parts:
        m = re.match(r"## (\d+)\. (.+)", p)
        if m:
            secs[int(m.group(1))] = p
    return secs


def code_blocks(section_text):
    """返回该节内全部 ```python 代码块内容(不含围栏)。"""
    return re.findall(r"```python\n(.*?)```", section_text, re.S)


def extract_summary_items(skel):
    """从骨架提取顶层 class/def + 描述（docstring 首行/行尾注释/邻近注释）。"""
    items = []
    lines = skel.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"(class|def) (\w+)", ln)
        if not m:
            continue
        name, desc = m.group(2), ""
        if i + 1 < len(lines) and '"""' in lines[i + 1]:
            desc = lines[i + 1].strip().strip('"').strip()
        if not desc:
            cm = re.search(r"#\s*(.+)$", ln)
            if cm:
                desc = cm.group(1).strip()
        if not desc:
            for j in range(i + 1, min(i + 7, len(lines))):
                if re.match(r"\S", lines[j]) and "def" not in lines[j][:5]:
                    break
                cm = re.search(r"#\s*(.+)$", lines[j])
                if cm:
                    desc = cm.group(1).strip()
                    break
                if '"""' in lines[j]:
                    desc = lines[j].strip().strip('"').strip()
                    break
        items.append((name, desc or name))
    return items


def flatten_json_keys(json_block):
    """把 JSON schema 块平铺为键路径列表。"""
    try:
        import json
        obj = json.loads(re.sub(r"\.\.\.", "0", json_block))
    except Exception:
        return []
    keys = []

    def walk(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{prefix}{k}"
                if isinstance(v, dict):
                    walk(v, p + ".")
                else:
                    keys.append(p)
        else:
            keys.append(prefix.rstrip("."))

    walk(obj)
    return keys


def rewrite_lab(num, md, dry):
    text = md.read_text(encoding="utf-8")
    secs = split_sections(text)
    if 2 not in secs:
        return None
    lab_name = md.stem
    code_dir = CODE / lab_name
    is_stage1 = num <= 5

    blocks = code_blocks(secs[2])
    skeleton = "\n\n".join(b for b in blocks if "assert" not in b and not re.search(r"# T\d", b))
    selftest = "\n\n".join(b for b in blocks if "assert" in b or re.search(r"# T\d", b))
    if not skeleton and blocks:
        skeleton = blocks[0]

    # ---- 生成/更新 code 骨架（stage1 已有项目，跳过文件生成）----
    wrote = []
    if not is_stage1 and not dry:
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "skeleton.py").write_text(
            f'"""Lab {num:02d} 接口骨架。TODO 注释即实现要求，讲义见 docs 对应章节。"""\n\n'
            + skeleton + "\n", encoding="utf-8")
        (code_dir / "selftest.py").write_text(
            f'"""Lab {num:02d} 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""\n'
            + ("import mlx.core as mx\n" if "mx." in selftest else "")
            + ("import math\n" if "math." in selftest else "")
            + "\n\n" + selftest + "\n\n\n"
            + "\n\n" + selftest + "\n\n\n"
            + "def check_implementation() -> None:\n"
            + '    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""' + "\n"
            + '    raise NotImplementedError("TODO: 实现讲义自测断言")' + "\n", encoding="utf-8")
        deps = STAGE_DEPS.get(num // 10 if num < 10 else int(str(num)[0]), [])
        (code_dir / "pyproject.toml").write_text(
            f'[project]\nname = "{lab_name}"\nversion = "0.1.0"\n'
            f'description = "实验 {num}: 见讲义 docs/{md.parent.name}/{md.name}"\n'
            f'requires-python = ">=3.11"\ndependencies = [\n'
            + "".join(f'    "{d}",\n' for d in deps) + "]\n", encoding="utf-8")
        (code_dir / "README.md").write_text(
            f"# Lab {num:02d}\n\n讲义：`docs/{md.parent.name}/{md.name}`\n\n"
            "- `skeleton.py` 接口骨架（TODO 即实现要求）\n"
            "- `selftest.py` 自测断言（先让它们变绿）\n"
            "- `run_lab{num:02d}.py` 实验入口（自测 -> 对照实验 E 系列 -> output/）\n\n"
            "```bash\nuv sync && uv run run_lab{num:02d}.py\n```\n\n"
            "观测指标落盘约定: 对照实验因变量写入 `output/metrics.json`（键名与讲义 §3/§5 一致），"
            "图表写入 `output/figures/`。\n", encoding="utf-8")
        # run 脚本: 从 §3 表格提取 E 系列编号生成占位
        es = sorted(set(re.findall(r"\| (E\d+) \|", secs.get(3, ""))))
        e_funcs = "\n\n".join(
            'def ' + e.lower() + '(metrics: dict) -> None:\n'
            '    """对照实验 ' + e + ': 讲义 §3 表格该行。"""\n'
            '    raise NotImplementedError("TODO: ' + e + '")' for e in es) or "pass"
        run_head = '"""Lab ' + f'{num:02d}' + ' 入口: selftest -> 对照实验 -> output/metrics.json + figures."""\n'
        (code_dir / f"run_lab{num:02d}.py").write_text(
            run_head
            + 'from selftest import check_implementation\n\n\n' + e_funcs + '\n\n\n'
            + 'def main() -> None:\n'
            + '    check_implementation()\n'
            + '    metrics: dict = {}\n'
            + ''.join('    ' + e.lower() + '(metrics)\n' for e in es)
            + '    import json\n    from pathlib import Path\n'
            + '    out = Path(__file__).parent / "output"\n'
            + '    out.mkdir(exist_ok=True)\n'
            + '    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))\n\n\n'
            + 'if __name__ == "__main__":\n    main()\n', encoding="utf-8")
        wrote = ["skeleton.py", "selftest.py", "pyproject.toml", "README.md",
                 f"run_lab{num:02d}.py"]

    # ---- 讲义 §2 改写 ----
    items = extract_summary_items(skeleton)
    sec2 = secs[2]
    intro = re.split(r"```python", sec2, maxsplit=1)[0]  # 引言段（代码块之前的正文）
    base = f"../../code/{lab_name}"
    if is_stage1:
        base = f"../../code/{lab_name}"
    listing = "\n".join(f"- `{name}` — {desc}" for name, desc in items)
    link_block = (
        f"\n实现清单（签名与 TODO 见骨架源码）：\n\n{listing}\n\n"
        f"- 接口骨架：[`{base}/skeleton.py`]({base}/skeleton.py)"
        if not is_stage1 else
        f"\n实现清单（模块与接口见源码）：\n\n{listing}\n\n"
        f"- 模块源码：[`code/{lab_name}/`]({base}/)"
    )
    if not is_stage1:
        link_block += (
            f"\n- 自测断言（红灯标准，实现完成前应全部失败）：[`{base}/selftest.py`]({base}/selftest.py)"
            f"\n- 实验入口：`cd code/{lab_name} && uv sync && uv run run_lab{num:02d}.py`")
    else:
        link_block += (
            f"\n- 自测断言（红灯标准）：`run_lab{num:02d}.py` 的 `check_implementation()`"
            f"\n- 实验入口：`cd code/{lab_name} && uv run run_lab{num:02d}.py`")
    new_sec2 = intro.rstrip() + "\n" + link_block + "\n\n"
    text = text.replace(sec2, new_sec2)

    # ---- §5 改写: JSON 块 -> 键路径清单 ----
    if 5 in secs:
        sec5 = secs[5]
        jb = re.findall(r"```json\n(.*?)```", sec5, re.S)
        if jb:
            keys = flatten_json_keys(jb[0])
            key_list = "\n".join(f"- `{k}`" for k in keys)
            head = re.split(r"```json", sec5, maxsplit=1)[0].rstrip()
            new_sec5 = (
                head.replace("（metrics.json）", "")
                + "\n\n以下指标在实验运行后写入 `output/metrics.json`（键名与 §3 对照实验表的因变量对应；"
                "写入逻辑见 `run_lab" + f"{num:02d}" + ".py`）：\n\n" + key_list + "\n\n")
            text = text.replace(sec5, new_sec5)

    if not dry:
        md.write_text(text, encoding="utf-8")
    return {"lab": lab_name, "items": len(items), "selftest": bool(selftest), "wrote": wrote}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="如 lab06-mlp-from-scratch")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for num, md in labs():
        if args.only and md.stem != args.only:
            continue
        r = rewrite_lab(num, md, args.dry)
        if r:
            print(("DRY " if args.dry else "") + f"{r['lab']}: 清单{r['items']}项 "
                  f"自测={r['selftest']} 生成={r['wrote'] or '(stage1 只瘦身)'}")


if __name__ == "__main__":
    main()
