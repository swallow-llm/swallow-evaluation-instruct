
from typing import Union, Optional
from vllm import LLM
from vllm.reasoning import ReasoningParserManager, ReasoningParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

# Sourced from vLLM unittest run_reasoning_extraction() method.
# Ref. https://github.com/vllm-project/vllm/blob/v0.9.1/tests/reasoning/utils.py#L35
def run_reasoning_extraction(model_output: str, reasoning_parser: str, 
                            vllm_engine: Optional[LLM] = None,
                            hf_tokenizer: Optional["transformers.PreTrainedTokenizer"] = None,
                            request: Union[ChatCompletionRequest, None] = None,
                            streaming: bool = False,
) -> tuple[Optional[str], Optional[str]]:
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
    
    reasoner_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser) 
    obj_reasoning_parser = reasoner_cls(tokenizer=tokenizer)
    if request is None:
        request = ChatCompletionRequest(messages=[], model=model)
    
    reasoning, content = run_reasoning_extraction_nonstreaming(
        obj_reasoning_parser, model_output, request)
    return reasoning, content
    
def run_reasoning_extraction_nonstreaming(
    reasoning_parser: ReasoningParser, model_output: str,
    request: Union[ChatCompletionRequest, None] = None,
) -> tuple[Optional[str], Optional[str]]:    
    request = request or ChatCompletionRequest(messages=[], model="test-model")    
    return reasoning_parser.extract_reasoning_content(model_output=model_output, request=request)