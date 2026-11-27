#!/usr/bin/env bash
# 统一构建入口 —— 编排书籍构建 / 主题安装与发布 / 本地预览 / 全量检查
#
# 用法:
#   ./build.sh                     # 构建（默认主题 genesis-book）
#   ./build.sh book <theme>        # 构建指定主题（genesis-book | paper-book | indigo-atlas）
#   ./build.sh install             # quarto add 重装线上主题（发布闭环验证）
#   ./build.sh release <msg>       # 主题发布：同步到发布仓库 + commit + push
#   ./build.sh serve [port]        # 本地预览 _book（默认 8765）
#   ./build.sh check               # 全量检查：构建校验 + 实验代码红灯
#   ./build.sh all                 # install + book（线上主题闭环构建）
set -euo pipefail
cd "$(dirname "$0")"

THEMES_REPO=~/my_proj/quarto-book-themes
GH_REPO=dontreadthisline/quarto-book-themes
THEMES=(genesis-book paper-book indigo-atlas)

C_G='\033[32m'; C_Y='\033[33m'; C_R='\033[31m'; C_0='\033[0m'
say()  { printf "${C_G}==>${C_0} %s\n" "$*"; }
warn() { printf "${C_Y}WARN:${C_0} %s\n" "$*"; }
die()  { printf "${C_R}ERROR:${C_0} %s\n" "$*" >&2; exit 1; }

cmd_book() {
  local theme="${1:-genesis-book}"
  [[ " ${THEMES[*]} " == *" $theme "* ]] || die "未知主题: $theme（可选: ${THEMES[*]}）"
  say "构建书籍 [theme=$theme]"
  GENESIS_THEME="$theme" ./book_render/scripts/build-book.sh
  say "完成: _book/index.html (open _book/index.html 或 ./build.sh serve)"
}

cmd_install() {
  say "quarto add 重装线上主题 ($GH_REPO)"
  rm -rf _extensions
  quarto add --no-prompt "$GH_REPO" 2>&1 | grep -viE '^\?' || die "安装失败（网络/代理？）"
  ls _extensions/dontreadthisline/ 2>/dev/null || die "扩展未落盘"
}

cmd_sync() {
  say "同步主题（主仓库 -> 发布仓库）"
  [[ -d "$THEMES_REPO" ]] || die "发布仓库不存在: $THEMES_REPO"
  for t in "${THEMES[@]}"; do
    [[ -d "book_render/themes/$t" ]] || continue
    mkdir -p "$THEMES_REPO/_extensions/dontreadthisline/$t"
    rsync -a --delete --exclude '.DS_Store' \
      "book_render/themes/$t/" "$THEMES_REPO/_extensions/dontreadthisline/$t/"
  done
}

cmd_release() {
  local msg="${1:-theme update}"
  cmd_sync
  say "发布: commit + push"
  (cd "$THEMES_REPO" && git add -A \
    && git diff --cached --quiet && warn "无变更，跳过推送" \
    || { git commit -qm "$msg" && git push -q; })
  say "已推送: https://github.com/$GH_REPO"
}

cmd_serve() {
  local port="${1:-8765}"
  [[ -d _book ]] || die "尚无 _book，先 ./build.sh"
  say "预览 http://localhost:$port (Ctrl-C 停止)"
  cd _book && exec python3 -m http.server "$port"
}

cmd_check() {
  say "检查 1/2: 构建校验（含 CDN/死链/本地 404）"
  ./book_render/scripts/build-book.sh > /dev/null
  say "检查 2/2: 实验代码红灯（预期 NotImplementedError）"
  local fail=0
  for d in code/lab*/; do
    local out; out=$(cd "$d" && uv run run_*.py 2>&1) || true
    if grep -q 'NotImplementedError' <<<"$out"; then
      printf "  ${C_G}PASS${C_0}  %s\n" "${d%/}"
    else
      printf "  ${C_R}FAIL${C_0}  %s（既非红灯也非预期输出）\n" "${d%/}"; fail=1
    fi
  done
  [[ $fail -eq 0 ]] && say "全部通过" || die "存在异常实验项目"
}

case "${1:-help}" in
  book)    shift; cmd_book "$@" ;;
  install) cmd_install ;;
  sync)    cmd_sync ;;
  release) shift; cmd_release "$@" ;;
  serve)   shift; cmd_serve "$@" ;;
  check)   cmd_check ;;
  all)     cmd_install && cmd_book indigo-atlas ;;
  help|*)  sed -n '2,12p' "$0" | sed 's/^# \{0,2\}//' ;;
esac
