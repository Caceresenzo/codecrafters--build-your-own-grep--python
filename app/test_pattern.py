import typing

import pytest

if typing.TYPE_CHECKING:
    from .regex import Pattern
else:
    from regex import Pattern


def test_literal():
    pattern = Pattern.compile(r"a")

    assert pattern.matcher("a").find(0) is True
    assert pattern.matcher("b").find(0) is False


def test_range_digits():
    pattern = Pattern.compile(r"\d")

    assert pattern.matcher("1").find(0) is True
    assert pattern.matcher("a").find(0) is False


def test_range_words():
    pattern = Pattern.compile(r"\w")

    assert pattern.matcher("a").find(0) is True
    assert pattern.matcher("?").find(0) is False


def test_positive_character_group():
    pattern = Pattern.compile(r"[abc]")

    assert pattern.matcher("a").find(0) is True
    assert pattern.matcher("d").find(0) is False


def test_positive_character_group_with_backslash():
    pattern = Pattern.compile(r"[abc\\]")

    assert pattern.matcher("\\").find(0) is True
    assert pattern.matcher("d").find(0) is False


def test_positive_character_group_with_digits():
    pattern = Pattern.compile(r"[abc\d]")

    assert pattern.matcher("a").find(0) is True
    assert pattern.matcher("1").find(0) is True
    assert pattern.matcher("d").find(0) is False


def test_negative_character_group():
    pattern = Pattern.compile(r"[^abc]")

    assert pattern.matcher("a").find(0) is False
    assert pattern.matcher("d").find(0) is True


def test_negative_character_group_with_digits():
    pattern = Pattern.compile(r"[^abc\d]")

    assert pattern.matcher("a").find(0) is False
    assert pattern.matcher("1").find(0) is False
    assert pattern.matcher("d").find(0) is True


def test_escape():
    pattern = Pattern.compile(r"\d\\\d\\\d apples")

    assert pattern.matcher("sally has 12 apples").find(0) is False


def test_start_anchor():
    pattern = Pattern.compile(r"^log")

    assert pattern.matcher("log").find(0) is True
    assert pattern.matcher("slog").find(0) is False


def test_end_anchor():
    pattern = Pattern.compile(r"dog$")

    assert pattern.matcher("dog").find(0) is True
    assert pattern.matcher("dogs").find(0) is False


def test_match_one_or_more_times():
    pattern = Pattern.compile(r"ca+ts")

    assert pattern.matcher("ca").find(0) is False
    assert pattern.matcher("cats").find(0) is True
    assert pattern.matcher("caats").find(0) is True
    assert pattern.matcher("caaats").find(0) is True
    assert pattern.matcher("dog").find(0) is False


def test_match_one_or_more_times_with_after():
    pattern = Pattern.compile(r"ca+at")

    assert pattern.matcher("caaats").find(0) is True


def test_match_zero_or_one_times():
    pattern = Pattern.compile(r"dogs?")

    assert pattern.matcher("dog").find(0) is True
    assert pattern.matcher("dogs").find(0) is True


def test_wildcard():
    pattern = Pattern.compile(r"d.g")

    assert pattern.matcher("dog").find(0) is True


def test_capture():
    pattern = Pattern.compile(r"a(b.)de")
    matcher = pattern.matcher("abcde")

    assert matcher.find(0) is True
    assert matcher.group(0) == "abcde"
    assert matcher.group(1) == "bc"


def test_branch():
    pattern = Pattern.compile(r"a(a|b)")

    assert pattern.matcher("aa").find(0) is True
    assert pattern.matcher("ab").find(0) is True
    assert pattern.matcher("ac").find(0) is False


def test_branch_inner():
    pattern = Pattern.compile(r"^I see (\d (cat|dog|cow)s?)$")

    matcher = pattern.matcher("I see 1 cat")
    assert matcher.find(0) is True
    assert matcher.group(0) == "I see 1 cat"
    assert matcher.group(1) == "1 cat"
    assert matcher.group(2) == "cat"

    assert pattern.matcher("I see 3 cats").find(0) is True
    assert pattern.matcher("I see 1 dog").find(0) is True
    assert pattern.matcher("I see 3 dogs").find(0) is True
    assert pattern.matcher("I see 1 cow").find(0) is True
    assert pattern.matcher("I see 3 cows").find(0) is True
    assert pattern.matcher("I see 1 rabbit").find(0) is False


def test_branch_inner_with_quantifier():
    pattern = Pattern.compile(r"^I see (\d (cat|dog|cow)s?(, | and )?)+$")

    matcher = pattern.matcher("I see 1 cat, 2 dogs and 3 cows")
    assert matcher.find(0) is True
    assert matcher.group(0) == "I see 1 cat, 2 dogs and 3 cows"
    assert matcher.group(1) == "3 cows"
    assert matcher.group(2) == "cow"
    assert matcher.group(3) == " and "


def test_capture_with_backtrack():
    pattern = Pattern.compile(r"_([^a]+),")

    assert pattern.matcher("_bbb, c").find(0) is True


def test_backreference_static():
    pattern = Pattern.compile(r"(cat) and \1")

    assert pattern.matcher("cat and cat").find(0) is True
    assert pattern.matcher("cat and dog").find(0) is False


def test_backreference_pattern():
    pattern = Pattern.compile(r"([abcd]+) is \1")

    assert pattern.matcher("abcd is abcd").find(0) is True


def test_multiple_backreference():
    pattern = Pattern.compile(r"(\d+) (\w+) squares and \1 \2 circles")

    assert pattern.matcher("3 red squares and 3 red circles").find(0) is True
    assert pattern.matcher("3 red squares and 4 red circles").find(0) is False


def test_nested_backreference():
    pattern = Pattern.compile(r"('(cat) and \2') is the same as \1")

    assert pattern.matcher("'cat and cat' is the same as 'cat and cat'").find(0) is True


def test_nested_backreference2():
    pattern = Pattern.compile(r"((c.t|d.g) and (f..h|b..d)), \2 with \3, \1")
    matcher = pattern.matcher("cat and fish, cat with fish, cat and fish")

    assert matcher.find(0) is True
    assert matcher.group(0) == "cat and fish, cat with fish, cat and fish"
    assert matcher.group(1) == "cat and fish"
    assert matcher.group(2) == "cat"
    assert matcher.group(3) == "fish"


def test_any_plus_at_start():
    pattern = Pattern.compile(r".+ar")

    assert pattern.matcher("carx").find(0) is True


def test_star():
    pattern = Pattern.compile(r"watermelon*")

    assert pattern.matcher("watermelon").find(0) is True
    assert pattern.matcher("watermelo").find(0) is True


def test_exactly_n_times():
    pattern = Pattern.compile(r"ca{3}t")

    assert pattern.matcher("caaat").find(0) is True
    assert pattern.matcher("caat").find(0) is False


def test_minimum_n_times():
    pattern = Pattern.compile(r"ca{2,}t")

    assert pattern.matcher("caat").find(0) is True
    assert pattern.matcher("caaaaat").find(0) is True
    assert pattern.matcher("cat").find(0) is False
