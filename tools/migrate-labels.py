#!/usr/bin/env python3
"""图内标注迁移：数学内容（公式/符号/变量/数字）→ mtex(LaTeX)，纯中文 → 保留手写 label。

标准 2：公式必须 LaTeX；装饰性中文可以手写。
处理对象: tools/figs-*.py 中的 s.label(...)/F.text(...) 调用。
用法: python3 tools/migrate-labels.py [--dry]
"""
import argparse
import re
from pathlib import Path

MATH_UNI = "ρσμΣπγεβαδθλΠΔ∈⊙⊗×→⇒‖√≤≥±≈≠∞²³⁰¹₂₁₃₄⁵₆₇₈₉′−∙·"
CJK = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"


STRONG = r"[=<>≤≥≈≠+\-^_~‖×⊙∈→Π∑√]|[ρσμΣπγεβαδθλΔ]|[A-Za-z0-9]\s*[（(]|[A-Za-z]\d|\d[A-Za-z]|\d+\.\d+|\.\d"


def is_math(s: str) -> bool:
    # 强数学特征: 运算/关系符、希腊、函数式括号、字母紧邻数字
    if re.search(STRONG, s):
        return True
    # 纯拉丁/数字串（无中文）: 'N(0,I)' 类函数式或数值 → 迁移; 'ReLU'/'S' 词汇 → 保留
    if not re.search(CJK, s):
        return bool(re.search(r"[()=+]|\.[0-9]|[0-9]\.[0-9]|\d+", s))
    return False


SYM = {
    "‖": r"\|", "×": r"\times", "·": r"\cdot", "∈": r"\in", "⊙": r"\odot",
    "→": r"\to", "⇒": r"\Rightarrow", "≤": r"\le", "≥": r"\ge", "±": r"\pm",
    "≈": r"\approx", "≠": r"\ne", "∞": r"\infty", "√": r"\sqrt",
    "ρ": r"\rho", "σ": r"\sigma", "μ": r"\mu", "Σ": r"\Sigma", "π": r"\pi",
    "γ": r"\gamma", "ε": r"\epsilon", "α": r"\alpha", "β": r"\beta",
    "δ": r"\delta", "θ": r"\theta", "λ": r"\lambda", "Δ": r"\Delta",
    "−": "-", "²": "^2", "³": "^3", "¹": "^1", "⁰": "^0",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
}


def latexify(s: str) -> str:
    # 中文段包 \text{}，其余符号替换
    parts = re.split(f"({CJK}+)", s)
    out = []
    for seg in parts:
        if not seg:
            continue
        if re.fullmatch(f"{CJK}+", seg):
            out.append(r"\text{" + seg + "}")
        else:
            for k, v in SYM.items():
                seg = seg.replace(k, v)
            # 中文标点也进 text（，。：（）等）
            seg = re.sub(r"([\u3000-\u303f\uff00-\uffef]+)", lambda m: r"\text{" + m.group(1) + "}", seg)
            out.append(seg)
    joined = " ".join(x for x in out if x)
    # 收尾：$$ 空内容保护、去掉孤立反斜杠空格
    joined = joined.replace("  ", " ").strip()
    return joined


CALL = re.compile("(\\b(?:s|F)\\.(?:label|text))\\(\\s*([^,]+),\\s*([^,]+),\\s*('(?:[^'\\\\]|\\\\.)*'|\\\"(?:[^\\\\\"]|\\\\.)*\\\")(.*)\\)")


def migrate(path: Path, dry):
    text = path.read_text(encoding="utf-8")
    changed = []

    def repl(m):
        head, x, y, raw, rest = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        content = raw[1:-1].replace("\\'", "'")
        if not is_math(content):
            return m.group(0)  # 纯中文保留
        latex = latexify(content)
        if not latex:
            return m.group(0)
        changed.append(content[:30])
        # 选项第三个参数原样保留（size/color/anchor 兼容）
        return f"F.mtex({x}, {y}, '" + latex.replace("'", "\\\\'") + f"'{rest})"

    new = CALL.sub(repl, text)
    if new != text and not dry:
        path.write_text(new, encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    total = 0
    for p in sorted(Path("tools").glob("figs-*.py")):
        ch = migrate(p, args.dry)
        total += len(ch)
        print(f"{p.name}: {len(ch)} 处迁移" + ("（DRY）" if args.dry else ""))
        for c in ch[:4]:
            print(f"   {c}")
    print(f"共 {total} 处")


if __name__ == "__main__":
    main()
