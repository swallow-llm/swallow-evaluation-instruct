#!/usr/bin/env bash
CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="$CONF_DIR/tasks_runtime.csv"

# key, col_index -> 対応セル
_field() {
  local key="$1"
  local output_col="$2"
  local match_col="${3:-1}"

  awk -F',' -v key="$key" -v mcol="$match_col" -v ocol="$output_col" '
    NR==1 {next}
    {for(i=1;i<=NF;i++) gsub(/^[ \t\r]+|[ \t\r]+$/, "", $i)}
    $mcol==key {print $ocol; found=1; exit}
    END{exit !found}                    # 見つからなければ exit 1
  ' "$CSV"
}

# 公開関数
all_tasks()    { awk -F',' 'NR>1 {gsub(/^[ \t\r]+|[ \t\r]+$/, "", $1); print $1}' "$CSV"; }
task_script()  { _field "$1" 2; }
task_result()  { _field "$1" 3; }
task_framework() { _field "$1" 4; }
hrt()          { _field "$2" "$( [ "$1" = node_q ] && echo 5 || echo 6 )"; }  # node_q 以外は全て node_f の時間となっていることに注意（つまり cpu ノードも node_f の時間で取っている）
walltime()     { _field "$2" "$( [ "$1" = rt_HG ] && echo 7 || echo 8 )"; }  # rt_HG 以外は全て rt_HF の時間となっていることに注意（つまり cpu ノードも rt_HF の時間で取っている）
script_result() { _field "$1" 3 2; }
script_task() { _field "$1" 1 2; }