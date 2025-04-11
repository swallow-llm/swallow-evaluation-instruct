import argparse
import os
import json
import glob
from datetime import datetime
from typing import Any

from aggregate_utils.white_lists import (
    JMMLU_SOCIAL_SCIENCES, JMMLU_HUMANITIES, JMMLU_STEM, JMMLU_OTHERS
)
from aggregate_utils.funcs import (
    pick, micro_average
)

# Aggregate Config
# - Each entry must have 'name', 'func', and 'target'. 'target' must have 'name' and 'is_category'.
# - Each entry may have 'func_args' matching its 'func'.
AGGREGATE_CONF = [
    {
        'name': 'jmmlu_social_sciences', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_SOCIAL_SCIENCES}, 
        'target': {'name': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'name': 'jmmlu_humanities', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_HUMANITIES}, 
        'target': {'name': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'name': 'jmmlu_stem', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_STEM}, 
        'target': {'name': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'name': 'jmmlu_other', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match', 'white_list': JMMLU_OTHERS}, 
        'target': {'name': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
    {
        'name': 'jmmlu', 
        'func': micro_average, 
        'func_args': {'metric_key': 'extractive_match'}, 
        'target': {'name': 'swallow|swallow_jmmlu|0', 'is_category': True}
    },
]


def main(model_name: str, raw_outputs_dir: str, aggregated_outputs_dir: str):
    # raw_outputs_dir 内のすべての JSON ファイルを取得する
    pattern = os.path.join(raw_outputs_dir, "*.json")
    all_results_files: list[str] = glob.glob(pattern)
    
    # 各結果ファイルからタスク名，メトリクス，必要時間，実行日時を抽出し集約する
    all_results_dicts: list[dict[str, Any]] = []
    for file_path in all_results_files:
        filename = os.path.basename(file_path)
        # ファイル名は "results_YYYY-MM-DDTHH-MM-SS.microsec.json" の形と仮定する
        try:
            # "results_" と ".json" を除きタイムスタンプ部分を取得
            timestamp_str = filename[len("results_"):-len(".json")]
            execution_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")
        except Exception as e:
            print(f"Skipping file {filename} due to datetime parsing error: {e}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            
        # 結果が対象のモデルか確認
        if data.get("config_general", {}).get("model_name") != model_name:
            continue
        
        config = data.get("config_general", {})
        # total_evaluation_time_secondes があればそれを使用，なければ start_time と end_time の差分で計算
        if "total_evaluation_time_secondes" in config:
            try:
                required_time = float(config["total_evaluation_time_secondes"])
            except Exception:
                required_time = None
        else:
            start_time = config.get("start_time")
            end_time = config.get("end_time")
            if start_time is not None and end_time is not None:
                required_time = end_time - start_time
            else:
                required_time = None

        # results 内の各タスク毎の情報を抽出
        config_tasks = data["config_tasks"]
        for task_key, metrics in data.get("results", {}).items():
            # task_key の形式例: "custom|swallow_jmmlu:abstract_algebra|0"
            # "|" で分割し、2 番目（subset）に "_average" が含まれるものや task_key=="all" はスキップする
            parts = task_key.split("|")
            if len(parts) >= 2:
                task_name, subset_name = parts[1].split(":")
                if subset_name == "_average":
                    continue
            else:
                if task_key == "all":
                    continue
                task_name = task_key
                subset_name = ""
            
            entry = {
                "task_key": task_key,
                "task": task_name,
                "subset": subset_name,
                "sample_num": config_tasks['|'.join(task_key.split("|")[:-1])].get("effective_num_docs", -1),
                "execution_datetime": execution_dt.isoformat(),
                "required_time": required_time,
                "metrics": metrics
            }
            all_results_dicts.append(entry)

    # 各タスクごとに最新のエントリのみを抽出する（キーは "task:subset"）
    latest_results = {}
    for entry in all_results_dicts:
        task_and_subset = f"{entry['task']}:{entry['subset']}"
        current_dt = datetime.fromisoformat(entry["execution_datetime"])
        if task_and_subset not in latest_results:
            latest_results[task_and_subset] = entry
        else:
            prev_dt = datetime.fromisoformat(latest_results[task_and_subset]["execution_datetime"])
            if current_dt > prev_dt:
                latest_results[task_and_subset] = entry

    # aggregated_results の作成
    # AGGREGATE_CONF で定義されている各項目に対して、集約関数を適用する
    aggregated_results = {
        "model": model_name,
        "results": {},
        "overall": "",
        "tasks": []
    }
    for conf in AGGREGATE_CONF:
        name = conf['name']
        func = conf['func']
        func_args = conf['func_args']
        target = conf['target']
        try:
            value = func(latest_results, target, **func_args)
            assert value is not None, f"Try calculating {name}, but received None."
        except Exception as e:
            print(f"Error processing {name}: {e}")
            value = -1
        aggregated_results["results"][name] = value
        aggregated_results["tasks"].append(name)
    # overall: すべての結果の値をカンマ区切りの文字列として格納
    aggregated_results["overall"] = ",".join(str(v) for v in aggregated_results["results"].values())

    # aggregated_outputs_dir が存在しなければ作成する
    os.makedirs(aggregated_outputs_dir, exist_ok=True)
    aggregated_filepath = os.path.join(aggregated_outputs_dir, "aggregated_results.json")
    
    # 結果を aggregated_filepath に書き出す
    with open(aggregated_filepath, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"Aggregated results saved to {aggregated_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="処理対象のモデル名")
    parser.add_argument("raw_outputs_dir", type=str, help="個別結果ファイルが存在するディレクトリのパス")
    parser.add_argument("aggregated_outputs_dir", type=str, help="集約結果を保存するディレクトリのパス")
    args = parser.parse_args()
    main(**vars(args))