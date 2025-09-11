import pytest

from lighteval.tasks.swallow.metrics_translation_japanese import (
    multi_prefix_extraction_function,
    wmt20_enja_translation_span_extractor,
    wmt20_jaen_translation_span_extractor,
)


@pytest.mark.parametrize(
    "text,prefixes,mode,expected",
    [
        # no matches => empty
        ("no prefix here", ["日本語:", "`日本語:", "```日本語:"], "first_match", []),
        # single prefix, first_match and last_match are same
        ("日本語: こんにちは", ["日本語:", "`日本語:", "```日本語:"], "first_match", ["こんにちは"]),
        ("日本語\n\n日本語: こんにちは", ["日本語:", "`日本語:", "```日本語:"], "last_match", ["こんにちは"]),
        ("日本語\n\n日本語: こんにちは\nありがとう", ["日本語:"], "last_match", ["こんにちは"]),
        # multiple prefixes, first_match picks first
        ("日本語: A\n`日本語: B\n```日本語: C", ["日本語:", "`日本語:", "```日本語:"], "first_match", ["A"]),
        # whitespace tolerance: leading spaces and no space after colon
        ("  日本語:Hello\n\t`日本語:World and me.", ["日本語:", "`日本語:", "```日本語:"], "first_match", ["Hello"]),
        # multiple prefixes, last_match picks last
        ("日本語: A\n`日本語: B\n```日本語: C", ["日本語:", "`日本語:", "```日本語:"], "last_match", ["A"]),
        # any_match collects all for the specific prefix; "日本語:" in this case.
        ("日本語: A\n日本語: B\n日本語: C\n", ["日本語:", "`日本語:", "```日本語:"], "any_match", ["A", "B", "C"]),
        
    ],
)
def test_multi_prefix_extraction_function(text, prefixes, mode, expected):
    result = multi_prefix_extraction_function(text=text, prefixes=prefixes, extraction_mode=mode)
    assert result == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        # basic Japanese prefix
        ("日本語: さようなら", ["さようなら"]),
        # backtick prefix
        ("`日本語: foo bar", ["foo bar"]),
        # fenced prefix
        ("```日本語: code snippet```", ["code snippet```"]),
        # multiple occurrences -> last_match
        ("日本語: first\n日本語: second", ["second"]),
        # no prefix => empty
        ("no prefix at all", []),
        # extended prefixes
        ("**日本語: 太字強調表示**", ["太字強調表示**"]),
        ("翻訳文: 翻訳された文です。\n改行です。", ["翻訳された文です。"]),
        ("訳文：another translated text", ["another translated text"]),
        ("和訳：yet another translated text", ["yet another translated text"]),
    ],
)
def test_wmt20_enja_translation_span_extractor(text, expected):
    assert wmt20_enja_translation_span_extractor(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        # basic English prefix
        ("English: hello", ["hello"]),
        # backtick prefix
        ("`English: world", ["world"]),
        # fenced prefix
        ("```English: code block```", ["code block```"]),
        # multiple occurrences -> last_match
        ("English: one\nEnglish: two", ["two"]),
        # no prefix => empty
        ("nothing to extract", []),
    ],
)
def test_wmt20_jaen_translation_span_extractor(text, expected):
    assert wmt20_jaen_translation_span_extractor(text) == expected
