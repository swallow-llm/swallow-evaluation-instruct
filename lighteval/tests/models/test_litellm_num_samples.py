import pytest

pytest.importorskip("litellm")

from litellm.utils import ModelResponse  # noqa: E402

from lighteval.models.litellm_model import LiteLLMClient  # noqa: E402
from lighteval.models.model_input import GenerationParameters  # noqa: E402
from lighteval.tasks.requests import GreedyUntilRequest  # noqa: E402


def make_client(calls):
    client = LiteLLMClient.__new__(LiteLLMClient)  # no network / tokenizer needed
    client.generation_parameters = GenerationParameters()
    client.CONCURENT_CALLS = 20
    client.tok_encode = lambda text: [0] * len(str(text))

    def fake_call_api(prompt, return_logits, max_new_tokens, num_samples, stop_sequence):
        calls.append((prompt, num_samples))
        return ModelResponse(choices=[{"index": i, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": f"{prompt}-{i}"}}
                                      for i in range(num_samples)])

    client._LiteLLMClient__call_api = fake_call_api
    return client


def make_request(task_name, context, num_samples):
    # same do_sample / use_logits / stop_sequence / generation_size, as every swallow task: one split
    return GreedyUntilRequest(task_name=task_name, sample_index=0, request_index=0, context=context,
                              metric_categories=[], stop_sequence=[], generation_size=None,
                              num_samples=num_samples, do_sample=True)


def test_each_request_gets_its_own_num_samples():
    calls = []
    client = make_client(calls)
    requests = [
        make_request("aime_N32", "short aime problem", 32),
        make_request("lcb", "a much longer livecodebench problem statement " * 5, 10),
        make_request("single", "medium length prompt here", 1),
    ]
    responses = client.greedy_until(requests)
    assert [len(r.result) for r in responses] == [32, 10, 1]
    assert sorted(n for _, n in calls) == [1, 10, 32]
