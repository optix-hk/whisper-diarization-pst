"""Regression tests for helpers.get_sentences_speaker_mapping.

bug-1: transcripts produced via server.py had spurious spaces inserted
between CJK characters. Root cause was helpers.py unconditionally
appending `wrd + " "` after every word.

bug-2: after bug-1, Latin words embedded in CJK audio came out as
"H e l l o  w o r l d" because faster-whisper emits per-character Latin
tokens and the bug-1 fix inserted a space between every pair of non-CJK
tokens. A second fix tried to use Whisper's leading-space word-boundary
marker, but ctc_forced_aligner forces split_size="char" for CJK
languages, producing standalone " " space tokens between Latin words —
not leading-space markers. The leading-space approach silently dropped
the standalone space, producing "helloworld".

Fix: _append_word now receives detected_language and branches on
split mode. Char-split (zh/ja): bare concatenation, relying on the
standalone " " token to carry word boundaries. Word-split (other
languages): space between every pair of tokens.
"""

from helpers import _append_word, get_sentences_speaker_mapping


def _w(word, speaker=0, start=0, end=100):
    return {"word": word, "speaker": speaker, "start_time": start, "end_time": end}


# ---------------------------------------------------------------------------
# _append_word unit cases — char-split (CJK languages)
# ---------------------------------------------------------------------------
def test_append_word_char_split_bare_concat():
    """Char-split: tokens are concatenated directly."""
    assert _append_word("你好", "世", "zh") == "你好世"


def test_append_word_char_split_standalone_space_appended():
    """Char-split: standalone " " is appended as-is."""
    assert _append_word("hello", " ", "zh") == "hello "


def test_append_word_char_split_space_then_latin():
    """Char-split: after " " token, next char is concatenated."""
    assert _append_word("hello ", "w", "zh") == "hello w"


def test_append_word_char_split_space_then_cjk():
    """Char-split: after " " token, CJK char is concatenated (space kept)."""
    assert _append_word("hello ", "世", "zh") == "hello 世"


def test_append_word_char_split_latin_continuation():
    """Char-split: bare Latin after Latin is concatenated."""
    assert _append_word("h", "e", "zh") == "he"


def test_append_word_char_split_first_token():
    """Char-split: empty text + first token returns the token."""
    assert _append_word("", "你", "zh") == "你"


def test_append_word_char_split_empty_word():
    """Char-split: empty word returns text unchanged."""
    assert _append_word("hello", "", "zh") == "hello"


# ---------------------------------------------------------------------------
# _append_word unit cases — word-split (non-CJK languages)
# ---------------------------------------------------------------------------
def test_append_word_word_split_gets_space():
    """Word-split: space inserted between tokens."""
    assert _append_word("hello", "world", "en") == "hello world"


def test_append_word_word_split_first_word():
    """Word-split: empty text + first token returns the token."""
    assert _append_word("", "hello", "en") == "hello"


def test_append_word_word_split_cjk_to_latin():
    """Word-split: CJK->Latin gets a space."""
    assert _append_word("你好", "world", "en") == "你好 world"


def test_append_word_word_split_latin_to_cjk():
    """Word-split: Latin->CJK gets a space."""
    assert _append_word("hello", "世界", "en") == "hello 世界"


# ---------------------------------------------------------------------------
# get_sentences_speaker_mapping end-to-end — bug-1 regression
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

    ssm = get_sentences_speaker_mapping(wsm, spk_ts, "zh")

    assert len(ssm) == 1
    text = ssm[0]["text"].strip()
    assert text == "你好世界", f"expected no spaces between CJK chars, got {text!r}"


def test_japanese_words_have_no_inter_character_spaces():
    """bug-1 also covers Japanese (Hiragana/Katakana/Kanji mix)."""
    wsm = [
        _w("こ", start=0, end=100),
        _w("ん", start=100, end=200),
        _w("に", start=200, end=300),
        _w("ち", start=300, end=400),
        _w("は", start=400, end=500),
        _w("世", start=500, end=600),
        _w("界", start=600, end=700),
    ]
    spk_ts = [(0, 700, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts, "ja")

    text = ssm[0]["text"].strip()
    assert text == "こんにちは世界", f"got {text!r}"


def test_latin_words_keep_spaces():
    """Control case: Latin transcripts (word-split) must get single spaces."""
    wsm = [
        _w("hello", start=0, end=100),
        _w("world", start=100, end=200),
    ]
    spk_ts = [(0, 200, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts, "en")

    text = ssm[0]["text"].strip()
    assert text == "hello world", f"Latin spacing regressed, got {text!r}"


# ---------------------------------------------------------------------------
# get_sentences_speaker_mapping end-to-end — bug-2 regression
# ---------------------------------------------------------------------------
def test_mixed_cjk_latin_char_level_tokens():
    """bug-2: Latin words emitted as per-char tokens by ctc_forced_aligner
    (char-split for CJK languages) must render as proper words.

    Token shape: CJK chars + Latin chars + standalone " " between Latin
    words. This is the actual token shape produced by ctc_forced_aligner
    when split_size="char" (forced for zh/ja).
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
        _w("h", start=1600, end=1700),
        _w("e", start=1700, end=1800),
        _w("l", start=1800, end=1900),
        _w("l", start=1900, end=2000),
        _w("o", start=2000, end=2100),
        _w(" ", start=2100, end=2150),
        _w("w", start=2150, end=2250),
        _w("o", start=2250, end=2350),
        _w("r", start=2350, end=2450),
        _w("l", start=2450, end=2550),
        _w("d", start=2550, end=2650),
    ]
    spk_ts = [(0, 2650, 0)]

    ssm = get_sentences_speaker_mapping(wsm, spk_ts, "zh")

    text = ssm[0]["text"].strip()
    assert text == "你好世界的英文是hello world", (
        f"bug-2 regression: expected proper Latin word spacing, got {text!r}"
    )



