# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import regex as re

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class StringBasedMarkupReasoningParser(ReasoningParser):
    """
    tokenizerに依存せず、think_start_expr, response_start_expr のみで
    reasoning_content を抽出するパーサ。
    """

    def __init__(self, think_start_expr: str, response_start_expr: str):
        # tokenizerはNoneでOK
        super().__init__(tokenizer=None)
        self.think_start_expr = think_start_expr
        self.response_start_expr = response_start_expr
        # 正規表現パターンを動的に構築
        self.reasoning_regex = re.compile(
            rf"{re.escape(self.think_start_expr)}(.*?){re.escape(self.response_start_expr)}(.*)",
            re.DOTALL
        )

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        think_start_expr, response_start_expr で囲まれた部分を reasoning_content、
        それ以降を content として抽出。
        """
        re_match = self.reasoning_regex.findall(model_output)
        if not re_match:
            return None, model_output
        reasoning_content, response_content = re_match[0]
        if not response_content:
            return reasoning_content, None
        return reasoning_content, response_content

    def extract_reasoning_content_streaming(self, *args, **kwargs):
        raise NotImplementedError(f"streaming mode is not supported in {self.__class__.__name__} parser.")
