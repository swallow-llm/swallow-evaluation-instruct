# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import logging
import time
import math
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List

import yaml
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
    GenerativeMultiturnResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import is_litellm_available
from lighteval.models.utils import replace_none_with_empty_string, replace_none_content_with_reasoning_content

logger = logging.getLogger(__name__)

if is_litellm_available():
    import litellm
    from litellm import encode
    from litellm.caching.caching import Cache
    from litellm.utils import ModelResponse
    from litellm import BadRequestError
    logger.info(f"LiteLLM default request timeout: {litellm.DEFAULT_REQUEST_TIMEOUT}")

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    # litellm.cache = Cache(type="disk")  # キャッシュ機能を無効化


def merge_model_responses(responses: List[ModelResponse]) -> ModelResponse:
    """
    複数の ModelResponse を統合し、choices を連結した新しい ModelResponse を返す。
    あわせて usage の completion_tokens, estimated_cost, total_tokens も合算する。
    """
    if not responses:
        return None
    
    merged_response = deepcopy(responses[0])
        
    for resp in responses[1:]:
        merged_response.choices.extend(resp.choices)
        if hasattr(merged_response, "usage") and hasattr(resp, "usage"):
            for key, value in resp.usage:
                if key == "completion_tokens":
                    merged_response.usage["completion_tokens"] += value
                    merged_response.usage["total_tokens"] += value
                elif key in ("estimated_cost", ):
                    merged_response.usage[key] += value
    
    return merged_response

@dataclass
class LiteLLMModelConfig:
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    generation_parameters: GenerationParameters = None

    def __post_init__(self):
        if self.generation_parameters is None:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]

        model = config["base_params"]["model_name"]
        provider = config["base_params"].get("provider", None)
        base_url = config["base_params"].get("base_url", None)
        api_key = config["base_params"].get("api_key", None)
        generation_parameters = GenerationParameters.from_dict(config)
        return cls(
            model=model,
            provider=provider,
            base_url=base_url,
            generation_parameters=generation_parameters,
            api_key=api_key,
        )


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config, env_config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.model = config.model
        self.provider = config.provider or config.model.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = int(os.getenv("LITELLM_CONCURRENT_CALLS", 20))  # 100 leads to hitting Anthropic rate limits

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.set_verbose = False

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens):
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if "o1" in self.model or "o3" in self.model:
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, 32000)
        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):
        """Make API call with retries."""

        max_n = getattr(self.generation_parameters, "max_n", None)

        # max_n < num_samples の場合は n=1 に設定してAPIをnum_samples回呼び出す
        if max_n is not None and max_n < num_samples:
            logger.warning(f"Number of parallel generations `n` will be set to 1, and the process will repeat {num_samples} times.")
            responses = []
            for _ in range(num_samples):
                resp = self._call_litellm_completion(
                    prompt=prompt,
                    return_logits=return_logits,
                    max_new_tokens=max_new_tokens,
                    stop_sequence=stop_sequence,
                    n=1
                )
                responses.append(resp)
            return merge_model_responses(responses)
        else:
            return self._call_litellm_completion(
                prompt=prompt,
                return_logits=return_logits,
                max_new_tokens=max_new_tokens,
                stop_sequence=stop_sequence,
                n=num_samples
            )

    def _call_litellm_completion(self, prompt, return_logits, max_new_tokens, stop_sequence, n):
        """
        litellm.completion 呼び出し・リトライ処理． self.__call_api() をリネームしただけ
        """
        for attempt in range(self.API_MAX_RETRY):
            try:
                stop_sequence = self._prepare_stop_sequence(stop_sequence)
                max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

                if return_logits and not self.provider == "openai":
                    logger.warning("Returning logits is not supported for this provider, ignoring.")

                # Prepare kwargs for completion call
                kwargs = {
                    "model": self.model,
                    "messages": prompt,
                    "logprobs": return_logits if self.provider == "openai" else None,
                    "base_url": self.base_url,
                    "n": n,
                    "caching": False,
                    "api_key": self.api_key,
                }
                if self.provider == "openai" and ("o1" in self.model or "o3" in self.model or "o4" in self.model):
                    logger.warning("O* models do not support temperature, top_p, stop sequence. Disabling.")
                else:
                    kwargs.update(self.generation_parameters.to_litellm_dict())

                # reasoning_effort をサポートするモデルのみ追加
                if litellm.supports_reasoning(model=self.model):
                    if getattr(self.generation_parameters, "reasoning_effort", None) is not None:
                        kwargs["reasoning_effort"] = self.generation_parameters.reasoning_effort
                        logger.info(f"Set reasoning_effort: {self.generation_parameters.reasoning_effort}")

                if kwargs.get("max_completion_tokens", None) is None:
                    kwargs["max_completion_tokens"] = max_new_tokens

                response = litellm.completion(**kwargs, timeout=litellm.DEFAULT_REQUEST_TIMEOUT)

                # If response content is null, replace with empty string
                if response is not None:
                    for choice in response.choices:
                        if choice.message.content is None:
                            logger.info("Response is empty, replacing with reasoning content.")
                            choice.message.content = replace_none_content_with_reasoning_content(choice.message)
                return response
            except litellm.BadRequestError as e:
                logger.error(f"BadRequestError in API call: {e}")
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return ModelResponse()
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return ModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        
        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"

        results = []
        n_concurrency = math.ceil(self.CONCURENT_CALLS / num_samples)
        with ThreadPoolExecutor(n_concurrency) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            contexts = [c.context for c in dataset]
            max_new_tokens = dataset[0].generation_size  # could be none
            return_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples
            stop_sequence = requests[0].stop_sequence

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response in responses:
                result: list[str] = [choice.message.content for choice in response.choices]
                # In empty responses, the model should return an empty string instead of None
                result = ["" if text is None else text for text in result]
                
                cur_response = GenerativeResponse(                    
                    result=result,
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    def greedy_until_multi_turn(
        self,
        requests: list[GreedyUntilMultiTurnRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeMultiturnResponse]:
        """
        マルチターンの会話に対して、greedy decodingを使用して応答を生成します。

        Args:
            requests (list[GreedyUntilMultiTurnRequest]): コンテキストと終了条件を含むリクエストのリスト
            override_bs (int, optional): 生成のバッチサイズを上書きする値。デフォルトはNone。

        Returns:
            list[GenerativeMultiturnResponse]: 生成された応答のリスト
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        results = []

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Greedy Multi Turn Generation",
            position=1,
            leave=False,
            disable=False,
        ):
            # TODO: self._call_api_parallelを使用して各ターンの出力を生成すれば、並列化により高速化できる。
            # ただ、入力によってtemperatureが異なる場合に対応できないため、現状はself._call_apiを使用している。
            # したがって、並列化を実現するには、まず入力単位でtemperatureを指定できるように実装を変更する必要がある。
            contexts = [c.context for c in dataset]
            max_new_tokens = dataset[0].generation_size  # could be none
            return_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples
            stop_sequence = requests[0].stop_sequence
            for idx in tqdm(range(len(contexts))):
                turn_results = []
                multi_turn_context_all = contexts[idx]
                for turn in range(len(multi_turn_context_all)):
                    tmp_turn = 0
                    multi_turn_context = multi_turn_context_all[turn]
                    for i in range(len(multi_turn_context)):
                        if "model_response" in multi_turn_context[i]["content"]:
                            multi_turn_context[i]["content"] = multi_turn_context[i]["content"].format(model_response=turn_results[tmp_turn][0])
                            tmp_turn += 1
                    temperature = requests[idx].temperature
                    if temperature is not None:
                        self.generation_parameters.temperature = temperature

                    if temperature is not None and temperature == 0 and num_samples > 1:
                        multi_turn_response = self.__call_api(multi_turn_context, return_logits, max_new_tokens, 1, stop_sequence)
                        gen_text = multi_turn_response.choices[0].message.content
                        turn_results.append([gen_text] * num_samples)
                    else:
                        tmp_results = []
                        for i in range(num_samples):
                            multi_turn_response = self.__call_api(multi_turn_context, return_logits, max_new_tokens, 1, stop_sequence)
                            gen_text = multi_turn_response.choices[0].message.content
                            tmp_results.append(gen_text)
                        turn_results.append(tmp_results)

                results.append(
                    GenerativeMultiturnResponse(
                        result=turn_results,
                        input_tokens=[],
                        generated_tokens=[],
                        truncated_tokens_count=0,
                        padded_tokens_count=0,
                    )
                )

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode(self, text: str):
        enc = encode(model=self.model, text=text)
        if hasattr(enc, "ids"):
            return enc.ids
        return enc

    def tok_encode(self, text: str | list[str]):
        if isinstance(text, list):
            if isinstance(text[0], list):
                toks = [[self._encode(t["content"]) for t in sublist] for sublist in text]
                toks = [tok for sublist in toks for tok in sublist]
            else:
                toks = [self._encode(t["content"]) for t in text]
                toks = [tok for tok in toks if tok]
            return toks
        return self._encode(text)

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError
