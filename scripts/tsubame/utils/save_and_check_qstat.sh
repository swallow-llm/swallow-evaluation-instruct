#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"

# 現在のジョブと以前のジョブの状態を取得
qstat_output=$(qstat -u $USER)
old_jobs=$(cat qstat_history.out)

# 現在のジョブについても以前のジョブについても情報がない時は終了
if [[ -z "$qstat_output" && -z "$old_jobs" ]]; then
  exit 0
fi

# qstatの返り値を一時ファイルに保存
echo "$qstat_output" > current_qstat.out

# 一時ファイルを解析してジョブIDと状態を取得
current_jobs=$(awk 'NR>2 {print $1, $5}' current_qstat.out)

# 結果を格納する変数を初期化
results=""
declare -A current_job_map

# 各ジョブについて情報を取得
while read -r job_id state; do
  job_info=$(qstat -j "$job_id")

  task_kind=$(echo "$job_info" | grep job_name | awk '{print $2}')
  model_name=$(echo "$job_info" |grep stderr_path_list | sed 's|.*results/\([^/]*\)/\([^/]*\).*|\1/\2|')
  slots=$(echo "$job_info" | grep parallel | awk '{print $5}')
  priority=$(echo "$job_info" | grep priority | awk '{print $2}')

  # タスク名とモデル名が空でない場合のみ結果を蓄積
  if [[ -n "$task_kind" && -n "$model_name" ]]; then
  
    # ノードの種類を判定
    resource_list=$(echo "$job_info" | grep "hard_resource_list")
    gn_mig_value=$(echo $resource_list | grep -oP 'gn_mig=\K[0-9]+')
    if [[ $gn_mig_value -eq 2 ]]; then
        node_kind="node_q"
    elif [[ $gn_mig_value -eq 8 ]]; then
        node_kind="node_f"
    else
        echo "Failed to identify node kind for ${job_id}"
        node_kind="_"
    fi

    results+="$job_id\t$state\t$node_kind\t$use_vllm\t$slots\t$priority\t$task_kind\t$model_name\n"
    current_job_map["$job_id"]="$state $node_kind $use_vllm $slots $priority $task_kind $model_name"
  fi
done <<< "$current_jobs"


# 動的にモデル名カラム幅を決定
term_width=$(tput cols)

# 各カラム幅の合計  カラム間スペース数(6)
fixed_width=$((10+8+8+8+10+35+6))
max_model=100
available=$(( term_width - fixed_width ))
(( available < 0 )) && available=0
if (( available > max_model )); then
  model_width=$max_model
else
  model_width=$available
fi

# 共通の printf フォーマット
fmt="%-10s %-8s %-8s %-8s %-10s %-35s %-*.*s\n"

# 出力のヘッダー
printf "$fmt"  "job_ID" "state" "node" "slots" "priority" "task" "$model_width" "$model_width" "model name"
printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'

# 前回のジョブの状態を読み込む
if [ -n "$old_jobs" ]; then
  # 前回のジョブIDが現在のジョブIDにない場合は終了したとみなす
  echo "$old_jobs" | while read -r old_job_id old_state old_node_kind old_slots old_priority old_task_kind old_model_name; do
    if [[ -z "${current_job_map[$old_job_id]}" ]]; then
      printf "$fmt" \
      "$old_job_id" "done" "$old_node_kind" "$old_slots" "$old_priority" "$old_task_kind" \
      "$model_width" "$model_width" "$old_model_name"
    fi
  done
fi

# 現在のジョブ情報を表示
echo -e "$results" | while read -r job_id state node_kind slots priority task_kind model_name; do
  if [[ -n "$job_id" ]]; then
    printf "$fmt" \
    "$job_id" "$state" "$node_kind" "$slots" "$priority" "$task_kind" \
    "$model_width" "$model_width" "$model_name"
  fi
done

# 現在のジョブ情報を保存
echo -e "$results" > qstat_history.out

# 一時ファイルを削除
rm current_qstat.out