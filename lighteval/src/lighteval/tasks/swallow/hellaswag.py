from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

from lighteval.metrics.dynamic_metrics import IndicesExtractionConfig, multilingual_extractive_match_metric


# Prompt template from Llama-3.2-3B-Instruct-evals: 
# https://huggingface.co/datasets/meta-llama/Llama-3.2-3B-Instruct-evals
# Rendered template example is as follows:
"""
Given the following incomplete context and four possible completions (A, B, C and D), select the best completion.
Incomplete context: Home and Garden: [header] How to paint basement stairs [title] Remove any carpet or overlaid material from your basement stairs. [step] Remove staples left from the carpet installation with pliers. Look over all areas of the stairs to find holes and deep scratches.
A: Get rid of any floating debris and knock out any plumbing fixtures, doors or fittings. Also be sure to remove any railings, cabinets, or sections attached to the basement above ground.
B: Pound on the stripped carpet with a hammer. In most cases, you'll encounter gentle taps caused by hammering along the floor.
C: Use putty or wood filler and a putty knife to fill in holes. If you have a cement staircase, you will want to fill holes with epoxy.
D: Remove any tread tiles or other fixtures that are covered. Keep the stairs cool so that water and moisture can flow freely in the stairs and help them to dry.
Your response should end with "The best completion is [the_letter]" where the [the_letter] is one of A, B, C or D.
"""


HELLASWAG_QUERY_TEMPLATE = """
Given the following incomplete context and four possible completions (A, B, C and D), select the best completion. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Incomplete context: {Question}

A: {A}
B: {B}
C: {C}
D: {D}
""".strip()


def remove_separator_from_ending(text: str) -> str:
    """
    It removes separator strings such as [title], [step], and [substeps] from the HellaSwag completion.

    Args:
        text (str): ending string. Specifically, `endings` column in the hellaswag dataset. Ref. https://huggingface.co/datasets/Rowan/hellaswag
    """    
    tup_separators = ("[substeps]", "[title]", "[step]")
    for separator in tup_separators:
        text = text.replace(separator, "")
    
    return text

# Extractive multiple choice accuracy metric
multi_choice_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

def hellaswag_prompt_fn(line, task_name: str = None):
    question = f"{line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}"
    choices = list(map(remove_separator_from_ending, line["endings"]))
    assert len(choices) == 4, "length != 4."
    gold_index = int(line["label"])
    query = HELLASWAG_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=None,
    )
    
hellaswag_generative = LightevalTaskConfig(
    name="hellaswag",
    suite=["swallow"],
    prompt_function=hellaswag_prompt_fn,
    hf_repo="hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[multi_choice_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)