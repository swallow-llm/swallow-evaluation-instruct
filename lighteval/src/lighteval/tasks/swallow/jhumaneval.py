import ast
from functools import partial
import json
from typing import Any, List, Tuple

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, SampleLevelMetric
from lighteval.tasks.extended.lcb.codegen_metrics import (
    codegen_metrics,
    extract_last_code_block,
)
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


# Query template
SWALLOW_JHUMANEVAL_QUERY_TEMPLATE = """
以下に与えられる未完成のコードスニペットを、ドックストリングに書かれている仕様や例を満たすように続きを実装し、完全な実装を ``` と ``` で囲んで出力してください。
ただし、与えられている関数名や変数名はそのまま使ってください。

```
{code_snipet}
```

""".lstrip()    # 末尾の2重改行は残す（distill-llamaで性能の著しい劣化を確認したため）


def swallow_jhumaneval_prompt_fn(line, task_name: str = "swallow_jhumaneval") -> Doc:
    # 1. get the prompt from the line
    query = SWALLOW_JHUMANEVAL_QUERY_TEMPLATE.format(code_snipet=line['prompt'])

    # 2. return the doc with the same format as lcb
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "fn_name": line['entry_point'],
            "task_id": line['task_id'],
            "check_fn": line['test'],
        },
    )


# Metric
def codegen_metric_passk(predictions: list[str], formatted_doc: Doc, k: int , **kwargs) -> float:
    """Estimates the Pass@k metric for the code generation task.
    Extract the code from each prediction, Runs it for each sample and generations,
    and computes the Pass@k over the outputs.
    (This is a modified version of codegen_metric in lcb)
    """
    # Extract generated code snippets
    generated_code_snippets = [[extract_last_code_block(pred) for pred in predictions]]
    evaluation_sample = {
        "fn_name": formatted_doc.specific["fn_name"],
        "check_fn": formatted_doc.specific["check_fn"],
        "task_id": formatted_doc.specific["task_id"],
    }
    # This is a list of lists because
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

    metrics, results = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[k],
        num_process_evaluate=32,    # node_q: 32 / node_f: 160
    )

    # Record the results in the formatted_doc
    formatted_doc.specific["extracted_predictions"] = generated_code_snippets[0]
    formatted_doc.specific["results"] = json.dumps(results[0])
    formatted_doc.specific["context"] = formatted_doc.ctx
    
    return metrics[f"pass@{k}"]


## Create metrics for different k values
NUM_SAMPLES = 10

codegen_metric_passk_1 = partial(codegen_metric_passk, k=1)
jhumaneval_pass_1 = SampleLevelMetric(
    metric_name=f"jhumaneval_pass@1:{NUM_SAMPLES}",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_metric_passk_1,
    corpus_level_fn=np.mean,
)

codegen_metric_passk_10 = partial(codegen_metric_passk, k=10)
jhumaneval_pass_10 = SampleLevelMetric(
    metric_name=f"jhumaneval_pass@10:{NUM_SAMPLES}",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_metric_passk_10,
    corpus_level_fn=np.mean,
)

## Register both metrics
extend_enum(Metrics, "jhumaneval_pass_1", jhumaneval_pass_1)
extend_enum(Metrics, "jhumaneval_pass_10", jhumaneval_pass_10) 


# Task table
jhumaneval = LightevalTaskConfig(
    name="swallow_jhumaneval",
    prompt_function=swallow_jhumaneval_prompt_fn,
    suite=["swallow"],
    hf_repo="kogi-jwu/jhumaneval",
    hf_subset="jhumaneval",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    trust_dataset=True,
    stop_sequence=[],
    metric=[Metrics.jhumaneval_pass_1, Metrics.jhumaneval_pass_10],
    version=0,
)   