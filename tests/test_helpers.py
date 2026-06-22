"""Regression tests for helpers.get_sentences_speaker_mapping.

bug-1: transcripts produced via server.py had spurious spaces inserted
between CJK characters. Root cause was helpers.py:381 unconditionally
appending `wrd + " "` after every word. These tests pin the CJK-aware
join behaviour so the regression cannot silently re-land.
"""

import pytest

from helpers import _append_word, _is_cjk, get_sentences_speaker_mapping


def _w(word, speaker=0, start=0, end=100):
    return {"word": word, "speaker": speaker, "start_time": start, "end_time": end}


# ---------------------------------------------------------------------------
# _is_cjk unit cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ch,expected",
    [
        ("你", True),      # CJK Unified
        ("好", True),      # CJK Unified
        ("世", True),      # CJK Unified
        ("界", True),      # CJK Unified
        ("あ", True),      # Hiragana
        ("ア", True),      # Katakana
        ("한", True),      # Hangul
        ("、", True),      # CJK punctuation
        ("。", True),      # CJK punctuation
        ("a", False),      # ASCII
        ("A", False),      # ASCII
        ("1", False),      # digit
        (".", False),      # ASCII punctuation
        (",", False),      # ASCII punctuation
        ("é", False),      # Latin extended
        (" ", False),      # space itself is not CJK
        ("", False),       # empty string is not CJK
    ],
)
def test_is_cjk(ch, expected):
    assert _is_cjk(ch) is expected


# ---------------------------------------------------------------------------
# _append_word unit cases
# ---------------------------------------------------------------------------
def test_append_word_latin_gets_space():
    assert _append_word("hello", "world") == "hello world"


def test_append_word_first_word_no_leading_space():
    assert _append_word("", "hello") == "hello"


def test_append_word_cjk_to_cjk_no_space():
    assert _append_word("你好", "世界") == "你好世界"


def test_append_word_cjk_to_latin_no_space():
    assert _append_word("你好", "world") == "你好world"


def test_append_word_latin_to_cjk_no_space():
    assert _append_word("hello", "世界") == "hello世界"


# ---------------------------------------------------------------------------
# get_sentences_speaker_mapping end-to-end — the actual bug-1 regression
# ---------------------------------------------------------------------------
def test_chinese_words_have_no_inter_character_spaces():
    """bug-1: Chinese transcript must not have spaces between CJK chars."""
    wsm = [
        _w("你", start=0, end=100),
        _w("好", start=100, end=200),
        _w("世", start=200, end=300),
        _w("界", start=300, end=400),
    ]
    spk_ts = [(0, 400, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    assert len(ssm) == 1
    text = ssm[0]["text"].strip()
    assert text == "你好世界", f"expected no spaces between CJK chars, got {text!r}"
    assert " " not in text, f"no spaces expected in CJK-only segment, got {text!r}"


def test_japanese_words_have_no_inter_character_spaces():
    """bug-1 also covers Japanese (Hiragana/Katakana/Kanji mix)."""
    wsm = [
        _w("こんにちは", start=0, end=300),
        _w("世界", start=300, end=600),
    ]
    spk_ts = [(0, 600, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "こんにちは世界", f"got {text!r}"
    assert " " not in text


def test_latin_words_keep_spaces():
    """Control case: Latin transcripts must still get single spaces."""
    wsm = [
        _w("hello", start=0, end=100),
        _w("world", start=100, end=200),
    ]
    spk_ts = [(0, 200, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "hello world", f"Latin spacing regressed, got {text!r}"


def test_cjk_latin_boundary_has_no_space():
    """bug-1 rule: skip space at CJK<->Latin boundaries too."""
    wsm = [
        _w("你好", start=0, end=200),
        _w("world", start=200, end=400),
    ]
    spk_ts = [(0, 400, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "你好world", f"CJK<->Latin boundary should have no space, got {text!r}"
