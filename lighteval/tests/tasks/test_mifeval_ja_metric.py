import json
import os
import pytest
from pprint import pprint

from lighteval.tasks.swallow.mifeval_ja.main import mifeval_ja_metric, mifeval_ja_prompt, agg_inst_level_acc
from lighteval.tasks.requests import Doc

RESOURCES = os.path.join(os.path.dirname(__file__), "../resources")

@pytest.mark.parametrize("strictness", ["strict", "loose"])
def test_mifeval_ja_metric_against_reference(strictness):
    # 入力データ
    with open(os.path.join(RESOURCES, "ja_input_data.jsonl"), encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]
    with open(os.path.join(RESOURCES, "ja_input_response_data_gpt-4o-2024-08-06.jsonl"), encoding="utf-8") as f:
        response_data = [json.loads(line) for line in f]
    # 期待出力
    ref_path = os.path.join(
        RESOURCES,
        "ja_input_response_data_gpt-4o-2024-08-06",
        f"eval_results_{strictness}.jsonl"
    )
    with open(ref_path, encoding="utf-8") as f:
        ref_results = [json.loads(line) for line in f]
    assert len(input_data) == len(response_data)

    # 各サンプルごとにテスト
    for idx, (input_item, response_item, ref_item) in enumerate(zip(input_data, response_data, ref_results)):
        # Doc生成
        doc = mifeval_ja_prompt(input_item)
        
        # strict/looseで分岐
        if strictness == "strict":
            metric_result = mifeval_ja_metric(predictions=[response_item["response"]], formatted_doc=doc)
            # strict: prompt_level_strict_acc, inst_level_strict_acc
            assert metric_result["prompt_level_strict_acc"] == int(ref_item["follow_all_instructions"]), f"strict prompt_level mismatch at idx={idx}"
            assert metric_result["inst_level_strict_acc"] == ref_item["follow_instruction_list"], f"strict inst_level mismatch at idx={idx}"
        else:
            metric_result = mifeval_ja_metric(predictions=[response_item["response"]], formatted_doc=doc)
            # loose: prompt_level_loose_acc, inst_level_loose_acc
            assert metric_result["prompt_level_loose_acc"] == int(ref_item["follow_all_instructions"]), f"loose prompt_level mismatch at idx={idx}"
            assert metric_result["inst_level_loose_acc"] == ref_item["follow_instruction_list"], f"loose inst_level mismatch at idx={idx}"



def test_mifeval_ja_metric_accuracy_aggregate():    
    # 入力データ
    # MODEL_ID = "gpt-4o-2024-08-06"    
    MODEL_ID = "google__gemma-3-12b-it"
    
    print(f"Model ID: {MODEL_ID}")
    with open(os.path.join(RESOURCES, "ja_input_data.jsonl"), encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]
    with open(os.path.join(RESOURCES, f"ja_input_response_data_{MODEL_ID}.jsonl"), encoding="utf-8") as f:
        response_data = [json.loads(line) for line in f]
    # metric_scores.json 読み込み
    with open(os.path.join(
        RESOURCES,
        f"ja_input_response_data_{MODEL_ID}",
        "metric_scores.json"
    ), encoding="utf-8") as f:
        ref_scores = json.load(f)

    # 各サンプルごとに strict/loose の metric_result を取得
    strict_prompt_level = []
    strict_inst_level = []
    loose_prompt_level = []
    loose_inst_level = []

    # accuracy 集計
    strict_prompt_acc = sum(strict_prompt_level) / len(strict_prompt_level)
    strict_inst_acc = agg_inst_level_acc(strict_inst_level)
    loose_prompt_acc = sum(loose_prompt_level) / len(loose_prompt_level)
    loose_inst_acc = agg_inst_level_acc(loose_inst_level)

    # 各 accuracy を metric_scores.json の値と比較（pytest.approx で許容誤差指定）
    assert strict_prompt_acc == pytest.approx(ref_scores["prompt_level_strict_acc"], abs=1e-4), f"strict prompt_level accuracy mismatch: {strict_prompt_acc} vs {ref_scores['prompt_level_strict_acc']}"
    assert strict_inst_acc == pytest.approx(ref_scores["inst_level_strict_acc"], abs=1e-4), f"strict inst_level accuracy mismatch: {strict_inst_acc} vs {ref_scores['inst_level_strict_acc']}"
    assert loose_prompt_acc == pytest.approx(ref_scores["prompt_level_loose_acc"], abs=1e-4), f"loose prompt_level accuracy mismatch: {loose_prompt_acc} vs {ref_scores['prompt_level_loose_acc']}"
    assert loose_inst_acc == pytest.approx(ref_scores["inst_level_loose_acc"], abs=1e-4), f"loose inst_level accuracy mismatch: {loose_inst_acc} vs {ref_scores['inst_level_loose_acc']}"
    
    pprint(ref_scores)
    d = {
        "prompt_level_strict_acc": strict_prompt_acc,
        "inst_level_strict_acc": strict_inst_acc,
        "prompt_level_loose_acc": loose_prompt_acc,
        "inst_level_loose_acc": loose_inst_acc
    }
    pprint(d)
