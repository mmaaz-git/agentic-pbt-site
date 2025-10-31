#!/usr/bin/env python3
"""
Property-based test using Hypothesis to demonstrate the pandas read_clipboard() bug
where the last line is lost when text doesn't end with a newline.
"""

from hypothesis import given, strategies as st, example, settings


@st.composite
def lines_strategy(draw, min_lines=2, max_lines=10):
    num_lines = draw(st.integers(min_value=min_lines, max_value=max_lines))
    lines = [
        draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\n\r'), min_size=1, max_size=50))
        for _ in range(num_lines)
    ]
    return lines


@example(["a\tb", "c\td", "e\tf"])
@example(["line1", "line2", "line3", "line4"])
@given(lines_strategy(min_lines=2, max_lines=10))
@settings(max_examples=200)
def test_line_count_preserved_without_trailing_newline(lines):
    text_without_newline = "\n".join(lines)
    text_with_newline = text_without_newline + "\n"

    # Simulate the pandas logic from line 98 in clipboards.py
    processed_without = text_without_newline[:10000].split("\n")[:-1][:10]
    processed_with = text_with_newline[:10000].split("\n")[:-1][:10]

    expected_lines = lines[:10]

    assert processed_with == expected_lines, (
        f"With trailing newline: expected {expected_lines}, got {processed_with}"
    )

    assert processed_without == expected_lines, (
        f"WITHOUT trailing newline: expected {expected_lines}, got {processed_without}. "
        f"Last line '{lines[-1]}' was lost! This is a bug."
    )


if __name__ == "__main__":
    # Run the test
    test_line_count_preserved_without_trailing_newline()