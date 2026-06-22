"""Regression tests for helpers.get_sentences_speaker_mapping.

bug-1: transcripts produced via server.py had spurious spaces inserted
between CJK characters. Root cause was helpers.py unconditionally
appending `wrd + " "` after every word.

bug-2: after bug-1, Latin words embedded in CJK audio came out as
"H e l l o  w o r l d" because faster-whisper emits per-character Latin
tokens and my bug-1 fix inserted a space between every pair of non-CJK
tokens. Fix: _append_word now respects Whisper's leading-space word-
boundary marker — tokens with a leading space start a new word, tokens
without one are continuations appended directly.
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
def test_append_word_continuation_token_no_space():
    """Token without leading space is a continuation — append directly."""
    assert _append_word("hello", "world") == "helloworld"


def test_append_word_new_word_latin_gets_space():
    """Token with leading space starts a new word — insert one space."""
    assert _append_word("hello", " world") == "hello world"


def test_append_word_first_word_no_leading_space():
    """First token ever with no leading space — just append."""
    assert _append_word("", "hello") == "hello"


def test_append_word_first_word_strips_leading_space():
    """First token ever WITH leading space — strip it, no separator."""
    assert _append_word("", " hello") == "hello"


def test_append_word_new_word_cjk_to_latin_no_space():
    """bug-1 rule: new word at CJK<->Latin boundary gets no space."""
    assert _append_word("你好", " world") == "你好world"


def test_append_word_new_word_latin_to_cjk_no_space():
    """bug-1 rule: new word at Latin<->CJK boundary gets no space."""
    assert _append_word("hello", " 世界") == "hello世界"


def test_append_word_empty_token_after_strip_is_noop():
    """A bare-space token strips to empty — no-op."""
    assert _append_word("hello", " ") == "hello"


# ---------------------------------------------------------------------------
# get_sentences_speaker_mapping end-to-end — bug-1 regression
# ---------------------------------------------------------------------------
def test_chinese_words_have_no_inter_character_spaces():
    """bug-1: Chinese transcript must not have spaces between CJK chars.

    CJK tokens from faster-whisper have no leading spaces (CJK doesn't use
    word spacing), so they're all continuations and append directly.
    """
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
    """Control case: Latin transcripts with leading-space word markers
    (the shape faster-whisper actually emits) must get single spaces."""
    wsm = [
        _w(" hello", start=0, end=100),
        _w(" world", start=100, end=200),
    ]
    spk_ts = [(0, 200, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "hello world", f"Latin spacing regressed, got {text!r}"


def test_cjk_latin_boundary_has_no_space():
    """bug-1 rule: skip space at CJK<->Latin boundaries.

    Latin token carries a leading space (new-word marker) but the CJK side
    suppresses the separator.
    """
    wsm = [
        _w("你好", start=0, end=200),
        _w(" world", start=200, end=400),
    ]
    spk_ts = [(0, 400, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "你好world", f"CJK<->Latin boundary should have no space, got {text!r}"


# ---------------------------------------------------------------------------
# get_sentences_speaker_mapping end-to-end — bug-2 regression
# ---------------------------------------------------------------------------
def test_mixed_cjk_latin_char_level_tokens():
    """bug-2: Latin words emitted as per-char tokens by faster-whisper +
    ctc_forced_aligner must render as proper words, not 'H e l l o  w o r l d'.

    Token shape: CJK chars (no leading space) + Latin chars (no leading
    space except word-initial tokens which carry ' ' per Whisper BPE).
    """
    wsm = [
        _w("你", start=0, end=200),
        _w("好", start=200, end=400),
        _w("世", start=400, end=600),
        _w("界", start=600, end=800),
        _w("的", start=800, end=1000),
        _w("英", start=1000, end=1200),
        _w("文", start=1200, end=1400),
        _w("是", start=1400, end=1600),
        # "Hello" as char-level tokens, no leading space on first char
        _w("H", start=1600, end=1700),
        _w("e", start=1700, end=1800),
        _w("l", start=1800, end=1900),
        _w("l", start=1900, end=2000),
        _w("o", start=2000, end=2100),
        # "world" as char-level tokens, first char carries leading space
        _w(" w", start=2100, end=2200),
        _w("o", start=2200, end=2300),
        _w("r", start=2300, end=2400),
        _w("l", start=2400, end=2500),
        _w("d", start=2500, end=2600),
    ]
    spk_ts = [(0, 2600, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "你好世界的英文是Hello world", (
        f"bug-2 regression: expected proper Latin word spacing, got {text!r}"
    )


def test_mixed_cjk_latin_whole_word_tokens():
    """bug-2 robustness: same sentence but Latin words arrive as whole-word
    tokens (also valid faster-whisper output). Must produce the same result."""
    wsm = [
        _w("你", start=0, end=400),
        _w("好", start=400, end=800),
        _w("世", start=800, end=1200),
        _w("界", start=1200, end=1600),
        _w(" Hello", start=1600, end=2100),
        _w(" world", start=2100, end=2600),
    ]
    spk_ts = [(0, 2600, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "你好世界Hello world", (
        f"whole-word Latin tokens should also render correctly, got {text!r}"
    )


def test_punctuation_attaches_to_previous_word():
    """Punctuation tokens (no leading space) are continuations — they attach
    to the previous word with no space, as expected."""
    wsm = [
        _w(" hello", start=0, end=100),
        _w(",", start=100, end=150),
        _w(" world", start=150, end=250),
    ]
    spk_ts = [(0, 250, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts)

    text = ssm[0]["text"].strip()
    assert text == "hello, world", f"punctuation should attach, got {text!r}"
