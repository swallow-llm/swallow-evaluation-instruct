# MIT License

# LiveCodeBench swallowカスタム版

import numpy as np
from aenum import extend_enum
from datasets import get_dataset_config_names

from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, SampleLevelMetric
from lighteval.tasks.extended.lcb.main import lcb_codegeneration_prompt_fn, codegen_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig

# metricだけサンプル数10で新規定義
lcb_swallow_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:10",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_metric,
    corpus_level_fn=np.mean,
)
extend_enum(Metrics, "swallow_lcb_codegen_metric", lcb_swallow_codegen_metric)

configs = get_dataset_config_names("livecodebench/code_generation_lite", trust_remote_code=True)

lcb_swallow_tasks = []
for subset in configs:
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        suite=["swallow"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=None,
        metric=[Metrics.lcb_swallow_codegen_metric],
        stop_sequence=[],
        trust_dataset=True,
        version=0,
    )
    lcb_swallow_tasks.append(task)
