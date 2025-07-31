
from typing import Union, Optional
from vllm import LLM
from vllm.reasoning import ReasoningParserManager, ReasoningParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from lighteval.models.utils import replace_none_with_empty_string

# Sourced from vLLM unittest run_reasoning_extraction() method.
# Ref. https://github.com/vllm-project/vllm/blob/v0.9.1/tests/reasoning/utils.py#L35
def run_reasoning_extraction(model_output: str, reasoning_parser: str, 
                            vllm_engine: Optional[LLM] = None,
                            hf_tokenizer: Optional["transformers.PreTrainedTokenizer"] = None,
                            request: Union[ChatCompletionRequest, None] = None,
                            streaming: bool = False,
                            replace_none_content_with_reasoning_content = True
) -> tuple[Optional[str], Optional[str]]:
    """
    モデル出力からreasoning（深い推論過程）とcontent（最終的な出力）を抽出します。

    Args:
        model_output (str): モデルの出力テキスト。
        reasoning_parser (str): 使用するReasoningParserの名前。
        vllm_engine (Optional[LLM]): vLLMエンジンインスタンス。Noneの場合はhf_tokenizerを使用。
        hf_tokenizer (Optional[transformers.PreTrainedTokenizer]): トークナイザー。Noneの場合はvllm_engineを使用。
        request (Optional[ChatCompletionRequest]): OpenAI互換のリクエストオブジェクト。省略時はデフォルト値。
        streaming (bool): ストリーミング出力をサポートするか（現状未対応）。
        replace_none_content_with_reasoning_content (bool): contentがNoneの場合、reasoning_contentで代用するか。

    Returns:
        tuple[Optional[str], Optional[str]]: (reasoning_content, content)
            reasoning_content: 推論過程の抽出結果
            content: 出力の抽出結果
    """
    if streaming:
        raise NotImplementedError(f"Improvised reasoning extractor does not support streaming output.")
    elif (vllm_engine is None) and (hf_tokenizer is None):
        raise ValueError(f"You must specify either vllm_engine or hf_tokenizer.")
    elif hf_tokenizer is not None:
        tokenizer = hf_tokenizer
        model = getattr(hf_tokenizer, "name_or_path", "dummy")
    elif vllm_engine is not None:
        tokenizer = vllm_engine.get_tokenizer()
        model = getattr(tokenizer, "name_or_path", "dummy")
    
    # ToDo: もしも非対応のモデルが現れた場合はここに reasoning-parser を追加してください．
    reasoner_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
    obj_reasoning_parser = reasoner_cls(tokenizer=tokenizer)
    if request is None:
        request = ChatCompletionRequest(messages=[], model=model)
    
    reasoning_content, content = run_reasoning_extraction_nonstreaming(
        obj_reasoning_parser, model_output, request)
    if replace_none_content_with_reasoning_content:
        if (content is None) and (reasoning_content is not None):
            content = reasoning_content
        reasoning_content = replace_none_with_empty_string(reasoning_content)
        content = replace_none_with_empty_string(content)
    
    return reasoning_content, content
    
def run_reasoning_extraction_nonstreaming(
    reasoning_parser: ReasoningParser, model_output: str,
    request: Union[ChatCompletionRequest, None] = None,
) -> tuple[Optional[str], Optional[str]]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")    
    return reasoning_parser.extract_reasoning_content(model_output=model_output, request=request)
