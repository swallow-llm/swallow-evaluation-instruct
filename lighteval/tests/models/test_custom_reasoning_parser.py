import pytest
from lighteval.models.vllm.utils import run_reasoning_extraction_nonstreaming
from lighteval.models.vllm.reasoning_parser import DeepseekR1MarkupParser

test_cases = [
    {
        "input": "<think>あいうえお\nこんにちは。</think>かきくけこ",
        "expected": { "reasoning_content": "あいうえお\nこんにちは。", "content": "かきくけこ" }
    },
    {
        "input": "<think>logic</think>answer",
        "expected": { "reasoning_content": "logic", "content": "answer" }
    },
    {
        "input": "<think>一行目\n二行目\n三行目</think>\n最終出力",
        "expected": { "reasoning_content": "一行目\n二行目\n三行目", "content": "\n最終出力" }
    },
    {
        "input": "<think></think>本文だけ",
        "expected": { "reasoning_content": "", "content": "本文だけ" }
    },
    {
        "input": "<think>最後に考える</think>",
        "expected": { "reasoning_content": "最後に考える", "content": None}
    },
    {
        "input": "これは普通の文章です（タグなし）",
        "expected": { "reasoning_content": None, "content": "これは普通の文章です（タグなし）" }
    },
    {
        "input": "<think>閉じがない",
        "expected": { "reasoning_content": None, "content": "<think>閉じがない" }
    },
    {
        "input": "</think>だけある",
        "expected": { "reasoning_content": None, "content": "</think>だけある" }
    },
    {
        "input": "<think>外<think>内</think>外2</think>本文",
        "expected": {'reasoning_content': '外<think>内', 'content': '外2</think>本文'}
    },
    {
        "input": "<think>A</think>本<think>B</think>後",
        "expected": { "reasoning_content": "A", "content": "本<think>B</think>後" }
    },
    {
        "input": "   \t<think>\n  A \n</think>\n 本文",
        "expected": { "reasoning_content": "\n  A \n", "content": "\n 本文" }
    },
    {
        "input": "<think role=\"chain-of-thought\">X</think>Y",
        "expected": { "reasoning_content": None, "content": "<think role=\"chain-of-thought\">X</think>Y" }
    },
    {
        "input": "<think >X</think>Y",
        "expected": { "reasoning_content": None, "content": "<think >X</think>Y" }
    },
    {
        "input": "<THINK>X</THINK>Y",
        "expected": { "reasoning_content": None, "content": "<THINK>X</THINK>Y" }
    },
    {
        "input": "<think>X</Think>Y",
        "expected": { "reasoning_content": None, "content": "<think>X</Think>Y" }
    },
    {
        "input": "<think/>Y",
        "expected": { "reasoning_content": None, "content": "<think/>Y" }
    },
    {
        "input": "＜think＞X＜/think＞Y",
        "expected": { "reasoning_content": None, "content": "＜think＞X＜/think＞Y" }
    },
    {
        "input": "\ufeff<think>A</think>B",
        "expected": { "reasoning_content": "A", "content": "B" }
    },
    {
        "input": "<think>行1\r\n行2</think>後",
        "expected": { "reasoning_content": "行1\r\n行2", "content": "後" }
    },
    {
        "input": "<think><notatag>中身</notatag></think>本文",
        "expected": { "reasoning_content": "<notatag>中身</notatag>", "content": "本文" }
    },
    {
        "input": "前置きテキスト<think>A</think>後",
        "expected": { "reasoning_content": "A", "content": "後" }
    },
    {
        "input": "",
        "expected": { "reasoning_content": None, "content": "" }
    },
    {
        "input": "   ",
        "expected": { "reasoning_content": None, "content": "   " }
    },
    {
        "input": "</think><think>BBB</think>",
        "expected": { "reasoning_content": "BBB", "content": None }
    },
    {
        "input": "<think>A</think></think>B",
        "expected": { "reasoning_content": "A", "content": "</think>B" }
    }
]

def _eq_expected(got_reasoning, got_content, expected):
    return (
        got_reasoning == expected.get("reasoning_content") and
        got_content == expected.get("content")
    )

@pytest.mark.parametrize("case", test_cases)
def test_deepseek_r1_markup_parser(case):
    parser = DeepseekR1MarkupParser()
    text = case["input"]
    expected = case["expected"]
    try:
        reasoning_content, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser=parser,
            model_output=text,
            request=None
        )
    except Exception as e:
        assert False, f"Exception: {e} for input={repr(text)}"
    assert _eq_expected(reasoning_content, content, expected), (
        f"input={repr(text)}\n"
        f"expected: {expected}\n"
        f"got     : {{'reasoning_content': {reasoning_content!r}, 'content': {content!r}}}"
    )
