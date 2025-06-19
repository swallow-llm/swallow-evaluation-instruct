#!/usr/bin/env bash
CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="$CONF_DIR/tasks_runtime.csv"

# key, col_index -> 対応セル
_field() {
  awk -F',' -v key="$1" -v col="$2" '
    NR==1 {next}
    {for(i=1;i<=NF;i++) gsub(/^[ \t\r]+|[ \t\r]+$/, "", $i)}
    $1==key {print $col; found=1; exit}
    END{exit !found}                    # 見つからなければ exit 1
  ' "$CSV"
}

# 公開関数
all_tasks()    { awk -F',' 'NR>1 {gsub(/^[ \t\r]+|[ \t\r]+$/, "", $1); print $1}' "$CSV"; }
task_script()  { _field "$1" 2; }
task_result()  { _field "$1" 3; }
task_framework() { _field "$1" 4; }
hrt()          { _field "$2" "$( [ "$1" = node_q ] && echo 5 || echo 6 )"; }