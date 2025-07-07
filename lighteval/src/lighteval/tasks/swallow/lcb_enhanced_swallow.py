# MIT License

# LiveCodeBench swallowカスタム強化版

from typing import Any
import numpy as np
from aenum import extend_enum
from datasets import get_dataset_config_names
import json
from markdown_it import MarkdownIt

from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.lighteval_task import Doc
from lighteval.tasks.extended.lcb.codegen_metrics import codegen_metrics, translate_private_test_cases


def prepare_prompt(line: dict[str, Any]) -> str:
    """
    LiveCodeBench prompt newline delimiter modified version
    """
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        query += f"\n\n```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
        query += "\n\n```python\n# YOUR CODE HERE\n```\n\n"
    return query


def lcb_aug_codegeneration_prompt_fn(line, task_name: str = "lcb_aug:codegeneration") -> Doc:
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )


def extract_last_code_block(model_output: str) -> str:
    """
    モデルの出力における最後のコードブロックを抽出する．
    ただし，文中に現れるコードブロック以外の ``` や，文法的に成立していないコードブロックは無視する．
    抽出に失敗したときは出力全体を返す．
    """
    md = MarkdownIt()
    tokens = md.parse(model_output)
    codes = [t.content for t in tokens if t.type == "fence"]
    
    return codes[-1] if len(codes) > 0 else model_output


def aug_codegen_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    """Estimates the Pass@1 metric for the code generation task.
    Extract the code from each prediction, Runs it for each sample and generations,
    and computes the Pass@1 over the outputs.
    """
    # Extract generated code snippets
    generated_code_snippets = [[extract_last_code_block(pred) for pred in predictions]]  # noqa: F841
    evaluation_sample = {  # noqa: F841
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }
    # This is a list of lists because
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]
    
    metrics, results = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[1],  # Only run for Pass@1
        num_process_evaluate=8,
    )
    
    # Save results in the formatted_doc
    formatted_doc.specific["extracted_predictions"] = generated_code_snippets[0]
    formatted_doc.specific["results"] = json.dumps(results[0])
    formatted_doc.specific["context"] = formatted_doc.ctx
    
    return metrics["pass@1"]



# metricだけサンプル数10で新規定義
lcb_aug_codegenmetric_pass_at_k_n10 = SampleLevelMetric(
    metric_name="codegen_pass@1:10",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=aug_codegen_metric,
    corpus_level_fn=np.mean,
)
extend_enum(Metrics, "lcb_aug_codegenmetric_pass_at_k_n10", lcb_aug_codegenmetric_pass_at_k_n10)

configs = get_dataset_config_names("livecodebench/code_generation_lite", trust_remote_code=True)

lcb_aug_swallow_tasks = []
for subset in configs:
    name = "lcb_aug:codegeneration" if subset == "v4_v5" else f"lcb_aug:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        suite=["swallow"],
        prompt_function=lcb_aug_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=None,
        metric=[Metrics.lcb_aug_codegenmetric_pass_at_k_n10],
        stop_sequence=[],
        trust_dataset=True,
        version=0,
    )
    lcb_aug_swallow_tasks.append(task)
