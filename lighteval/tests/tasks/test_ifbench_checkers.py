"""Regression tests for IFBench checkers aligned with allenai/IFBench@4889186."""

from lighteval.tasks.extended.ifbench import instructions as I


def test_repeat_change_compares_words():
    c = I.RepeatChangeChecker("repeat:repeat_change")
    c.build_description(prompt_to_repeat="Write a python program to implement a simple JPEG compression algorithm.")
    assert c.check_following("Create a python program to implement a simple JPEG compression algorithm.")
    # the first word must differ and every other word must be kept
    assert not c.check_following("Create a program to implement a simple JPEG compression algorithm.")
    assert not c.check_following("Write a python program to implement a simple JPEG compression algorithm.")


def test_repeat_span_is_inclusive_and_keeps_index_zero():
    c = I.RepeatSpanChecker("repeat:repeat_span")
    c.build_description(prompt_to_repeat="a b c d e f g", n_start=2, n_end=4)
    assert c.check_following("c d e")
    assert not c.check_following("c d")
    for _ in range(5):
        z = I.RepeatSpanChecker("repeat:repeat_span")
        z.build_description(prompt_to_repeat="a b c d e f g h i j k l m n o", n_start=0, n_end=3)
        assert z._n_start == 0
        assert z.check_following("a b c d")


def test_words_position_ignores_punctuation():
    c = I.WordsPositionChecker("words:words_position")
    c.build_description(keyword="whisper")
    assert c.check_following("The whisper of the wind carries a soft Whisper today.")


def test_keyword_specific_position_ignores_punctuation_and_case():
    c = I.KeywordSpecificPositionChecker("words:keywords_specific_position")
    c.build_description(keyword="giggle", n=1, m=3)
    assert c.check_following("Yes, we giggle at the clouds. Then we rest.")


def test_european_capitals_sort_includes_kyiv():
    c = I.EuropeanCapitalsSortChecker("custom:european_capitals_sort")
    c.build_description()
    assert c.check_following(
        "Reykjavik, Helsinki, Oslo, Tallinn, Stockholm, Riga, Moscow, Copenhagen, Vilnius, Minsk, Dublin, Berlin, "
        "Amsterdam, Warsaw, London, Brussels, Kyiv, Prague, Luxembourg, Paris, Vienna, Bratislava, Budapest, Vaduz, "
        "Chisinau, Bern, Ljubljana, Zagreb"
    )


def test_pronoun_count_includes_demonstratives():
    c = I.PronounCountChecker("count:pronouns")
    c.build_description(N=3)
    assert c.check_following("This is what everyone wanted.")


def test_ngram_overlap_uses_word_trigrams():
    c = I.NGramOverlapChecker("ratio:overlap")
    # word trigrams of the reference: (a b c) (b c d) (c d e) (d e f); the response shares 2 of its 4 trigrams
    c.build_description(reference_text="a b c d e f", percentage=50)
    assert c.check_following("a b c d x y")
    c.build_description(reference_text="a b c d e f", percentage=100)
    assert c.check_following("a b c d e f")
    assert not c.check_following("a b c d x y")
