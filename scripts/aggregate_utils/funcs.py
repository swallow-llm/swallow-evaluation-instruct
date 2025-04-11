# Aggregate functions
# - must have latest_results and target as the 1st arg and 2nd arg.
# - may have three or more args followed.

def pick(latest_results: dict, target: dict, metric_key: str) -> float:
    """
    対象タスクの最新エントリから、指定された metric の値をそのまま返す．
    対象エントリが存在しなければ -1 を返す．
    """
    target_name = target.get("name")
    for entry in latest_results.values():
        if entry["task_key"] == target_name:
            return entry["metrics"].get(metric_key, -1)
    return -1


def micro_average(latest_results: dict, target: dict, metric_key: str, white_list: list[str]=[]) -> float:
    """
    対象タスクの全サブセットについて、指定された metric のサンプル数重み付き平均を計算する．
    もし対象となるエントリが無ければ -1 を返す．
    なお white_list に特定のサブセット群を渡すことで，サブセットの中でも計算の対象をフィルタリングすることができる．
    """
    target_name = target.get("name")
    total_sample = 0
    weighted_sum = 0.0
    for entry in latest_results.values():
        # task_key の例: "custom|swallow_jmmlu:public_relations|0" や "custom|swallow_jmmlu:abstract_algebra|0"
        parts = entry["task_key"].split("|")
        if len(parts) < 3:
            continue
        # parts[1] を ":" で分割して先頭部分を抽出し，base_key を作る．
        base_second, subset_name = parts[1].split(":")
        base_key = f"{parts[0]}|{base_second}|{parts[2]}"
        if (base_key == target_name) and ((len(white_list)==0) or (subset_name in white_list)):
            sample = entry.get("sample_num", 0)
            metric_value = entry["metrics"].get(metric_key)
            if metric_value is None:
                continue
            total_sample += sample
            weighted_sum += sample * metric_value
    if total_sample > 0:
        return weighted_sum / total_sample
    else:
        return -1