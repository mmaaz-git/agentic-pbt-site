#!/usr/bin/env python3
"""Test the clipboard bug reported for pandas.io.clipboard.read_clipboard()"""

from hypothesis import given, strategies as st, example, settings
import pandas as pd


# First, let's reproduce the exact bug that was reported
def test_simple_reproduction():
    """Simple reproduction test from the bug report"""
    print("\n=== Simple Reproduction Test ===")

    text_with_newline = "a\tb\nc\td\ne\tf\n"
    text_without_newline = "a\tb\nc\td\ne\tf"

    # This simulates what happens in read_clipboard at line 98
    processed_with = text_with_newline[:10000].split("\n")[:-1][:10]
    processed_without = text_without_newline[:10000].split("\n")[:-1][:10]

    print(f"Text with newline: {repr(text_with_newline)}")
    print(f"Processed with newline: {processed_with}")
    print(f"Expected: ['a\\tb', 'c\\td', 'e\\tf']")

    print(f"\nText without newline: {repr(text_without_newline)}")
    print(f"Processed without newline: {processed_without}")
    print(f"Expected: ['a\\tb', 'c\\td', 'e\\tf']")

    # Check the assertions
    try:
        assert processed_with == ["a\tb", "c\td", "e\tf"], f"With newline failed: {processed_with}"
        print("✓ With newline: PASSED")
    except AssertionError as e:
        print(f"✗ With newline: FAILED - {e}")

    try:
        assert processed_without == ["a\tb", "c\td", "e\tf"], f"Without newline failed: {processed_without}"
        print("✓ Without newline: PASSED")
    except AssertionError as e:
        print(f"✗ Without newline: FAILED - {e}")
        print("BUG CONFIRMED: Last line is lost when text doesn't end with newline")


# Now let's run the hypothesis test from the bug report
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
@settings(max_examples=50)  # Reduced for quick testing
def test_line_count_preserved_without_trailing_newline(lines):
    text_without_newline = "\n".join(lines)
    text_with_newline = text_without_newline + "\n"

    processed_without = text_without_newline[:10000].split("\n")[:-1][:10]
    processed_with = text_with_newline[:10000].split("\n")[:-1][:10]

    expected_lines = lines[:10]

    assert processed_with == expected_lines, (
        f"With trailing newline: expected {expected_lines}, got {processed_with}"
    )

    # This assertion will fail, demonstrating the bug
    assert processed_without == expected_lines, (
        f"WITHOUT trailing newline: expected {expected_lines}, got {processed_without}. "
        f"Last line '{lines[-1] if lines else ''}' was lost! This is a bug."
    )


def test_edge_cases():
    """Test various edge cases"""
    print("\n=== Edge Cases Test ===")

    test_cases = [
        ("single_line", "hello world"),
        ("two_lines_with_newline", "line1\nline2\n"),
        ("two_lines_without_newline", "line1\nline2"),
        ("empty_last_line", "line1\n\n"),
        ("tabs_with_newline", "col1\tcol2\nval1\tval2\n"),
        ("tabs_without_newline", "col1\tcol2\nval1\tval2"),
    ]

    for name, text in test_cases:
        lines = text[:10000].split("\n")[:-1][:10]
        print(f"{name}: input={repr(text)}, lines={lines}")


def analyze_split_behavior():
    """Analyze Python's split behavior to understand the bug"""
    print("\n=== Analyzing Split Behavior ===")

    examples = [
        "a\nb\nc\n",
        "a\nb\nc",
        "single",
        "single\n",
        "",
        "\n"
    ]

    for text in examples:
        split_result = text.split("\n")
        after_slice = text.split("\n")[:-1]
        print(f"Text: {repr(text)}")
        print(f"  split('\\n'): {split_result}")
        print(f"  split('\\n')[:-1]: {after_slice}")
        print()


if __name__ == "__main__":
    # Run simple reproduction
    test_simple_reproduction()

    # Run edge cases
    test_edge_cases()

    # Analyze split behavior
    analyze_split_behavior()

    # Run hypothesis test
    print("\n=== Running Hypothesis Test ===")
    try:
        test_line_count_preserved_without_trailing_newline()
        print("All hypothesis tests passed (unexpected!)")
    except AssertionError as e:
        print(f"Hypothesis test failed as expected: {str(e)[:200]}...")
        print("BUG CONFIRMED via property-based testing")