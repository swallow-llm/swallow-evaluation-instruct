# Aggregate functions
# - must have latest_results and target as the 1st arg and 2nd arg.
# - may have three or more args followed.

from functools import wraps

def resolve_multi_task_key(func):
    """
    target["task_key"] が "key1,key2,..." のように複数書かれている場合に
    ・latest_results で見つかったキーが 0 個なら -1 を返す
    ・1 個ならそのキーだけを target にセットして元関数へ委譲
    ・2 個以上見つかったら ValueError を送出
    単一キーの場合は何もしないでそのまま元関数へ委譲する
    """
    @wraps(func)
    def wrapper(latest_results: dict, target: dict, *args, **kwargs):
        raw_key = target.get("task_key", "")
        # ① task_key が空／単一ならそのまま
        if (not raw_key) or ("," not in raw_key):
            return func(latest_results, target, *args, **kwargs)

        # ② 複数キーを個別に試す
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        found_keys = []
        result_for_found = None

        for k in keys:
            ## keys から一つ選んで task_key としてセットし，元関数を走らせる 
            ## 見つからないと -1 が返る仕様を活用して，見つかったキーとその計算結果を保存しておく
            tmp_target = dict(target)
            tmp_target["task_key"] = k
            r = func(latest_results, tmp_target, *args, **kwargs)
            if r != -1:
                found_keys.append(k)
                result_for_found = r

        # ③ 見つかった結果に応じた処理
        if len(found_keys) == 0:
            ## キーが一つも見つからなかった場合は -1 を返す
            return -1

        elif len(found_keys) == 1:
            ## キーが一つだけ見つかった場合はその結果を採用
            return result_for_found

        else:
            ## キーが複数見つかった場合は ValueError を送出
            raise ValueError(
                f"Multiple task_keys {found_keys} are present in latest_results "
                f"for requested '{raw_key}'. Please disambiguate."
            )

    return wrapper


@resolve_multi_task_key
def pick(latest_results: dict, target: dict, metric_key: str) -> float:
    """
    対象タスクの最新エントリから、指定された metric の値をそのまま返す．
    対象エントリが存在しなければ -1 を返す．
    """
    task_key = target.get("task_key")
    for entry in latest_results.values():
        if entry["task_key"] == task_key:
            return entry["metrics"].get(metric_key, -1)
    print(f"Warning: {task_key} does not have {metric_key}.")
    return -1


@resolve_multi_task_key
def micro_average(latest_results: dict, target: dict, metric_key: str, white_list: list[str]=[]) -> float:
    """
    対象タスクの全サブセットについて、指定された metric のサンプル数重み付き平均を計算する．
    もし対象となるエントリが無ければ -1 を返す．
    なお white_list に特定のサブセット群を渡すことで，サブセットの中でも計算の対象をフィルタリングすることができる．
    """
    task_key = target.get("task_key")
    display_name = target.get("display_name")
    total_sample = 0
    weighted_sum = 0.0
    for entry in latest_results.values():
        # task_key の例: "swallow|swallow_jmmlu:public_relations|0" や "swallow|swallow_jmmlu:abstract_algebra|0"
        parts = entry["task_key"].split("|")
        if len(parts) < 3:
            continue
        # parts[1] を ":" で分割して先頭部分を抽出し，base_key を作る．
        if ":" in parts[1]:
            base_second, subset_name = parts[1].split(":", 1)
            base_key = f"{parts[0]}|{base_second}|{parts[2]}"
        else:
            base_second = parts[1]
            subset_name = ""
            base_key = f"{parts[0]}|{base_second}|{parts[2]}"

        if (base_key == task_key) and ((len(white_list)==0) or (subset_name in white_list)):
            sample_num = entry.get("sample_num", 0)
            metric_value = entry["metrics"].get(metric_key)
            if metric_value is None:
                print(f"Warning: {task_key} does not have {metric_key}.")
                continue
            total_sample += sample_num
            weighted_sum += sample_num * metric_value
    if total_sample > 0:
        return weighted_sum / total_sample
    else:
        return -1


@resolve_multi_task_key
def average_in_one_task(latest_results: dict, target: dict, metric_key_list: list[str]) -> float:
    """
    対象タスクに対して、指定されたmetric間の平均を計算する
    """
    task_key = target.get("task_key")
    result = 0
    count = 0
    for entry in latest_results.values():
        if entry["task_key"] == task_key:
            count += 1
            for metric_key in metric_key_list:
                assert metric_key in entry["metrics"], f"{metric_key} is not in {entry['metrics']}"
                result += entry["metrics"][metric_key]
    if count == 0:
        return -1

    return result / len(metric_key_list)
