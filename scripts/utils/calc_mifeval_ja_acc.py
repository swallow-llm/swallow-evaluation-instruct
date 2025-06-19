import argparse
import json
import os
import sys

from lighteval.tasks.swallow.mifeval_ja.main import mifeval_ja_prompt, mifeval_ja_metric

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def main():
    parser = argparse.ArgumentParser(description="日本語M-IFEval デバッグ用スクリプト．公式M-IFEvalで作った出力文を採点できます．")
    parser.add_argument("--input_data", required=True, help="M-IFEval同梱の ja_input_data.jsonl のパス")
    parser.add_argument("--response_data", required=True, help="公式M-IFEvalで作った出力文データ (jsonl) のパス")
    parser.add_argument("--output", default=None, help="採点結果を書き込む jsonl ファイル")
    args = parser.parse_args()

    # 入力データ読み込み
    with open(args.input_data, encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]
    with open(args.response_data, encoding="utf-8") as f:
        response_data = [json.loads(line) for line in f]

    if len(input_data) != len(response_data):
        print(f"入力数が一致しません: {len(input_data)} vs {len(response_data)}", file=sys.stderr)
        sys.exit(1)

    strict_prompt_level = []
    strict_inst_level = []
    loose_prompt_level = []
    loose_inst_level = []

    for input_item, response_item in zip(input_data, response_data):
        doc = mifeval_ja_prompt(input_item)
        metric_result = mifeval_ja_metric(predictions=[response_item["response"]], formatted_doc=doc)
        strict_prompt_level.append(metric_result["prompt_level_strict_acc"])
        strict_inst_level.append(metric_result["inst_level_strict_acc"])
        loose_prompt_level.append(metric_result["prompt_level_loose_acc"])
        loose_inst_level.append(metric_result["inst_level_loose_acc"])

    # accuracy 集計
    strict_prompt_acc = sum(strict_prompt_level) / len(strict_prompt_level)
    strict_inst_acc = sum(flatten(strict_inst_level)) / len(flatten(strict_inst_level))
    loose_prompt_acc = sum(loose_prompt_level) / len(loose_prompt_level)
    loose_inst_acc = sum(flatten(loose_inst_level)) / len(flatten(loose_inst_level))

    results = {
        "input": args.response_data,
        "prompt_level_strict_acc": strict_prompt_acc,
        "inst_level_strict_acc": strict_inst_acc,
        "prompt_level_loose_acc": loose_prompt_acc,
        "inst_level_loose_acc": loose_inst_acc,
    }

    print("==== mifeval_ja accuracy ====")
    for k, v in results.items():
        if isinstance(v, str):
            print(k)
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")

    # 出力ファイルに追記
    if args.output:
        out_path = args.output
        # 1行で追記
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(results, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
