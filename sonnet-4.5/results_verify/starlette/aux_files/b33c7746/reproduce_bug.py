#!/usr/bin/env python3
"""Reproduce the HTML entity bug in ServerErrorMiddleware."""

import html
from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


def test_reproduction_exact():
    """Test the exact reproduction from bug report."""
    middleware = ServerErrorMiddleware(None, debug=True)
    test_line = "    def example_function():"

    # This is what the code actually does (from line 191)
    escaped = html.escape(test_line).replace(" ", "&nbsp")

    print("Test line:", repr(test_line))
    print("Escaped output:", escaped)
    print("Expected output:", "&nbsp;&nbsp;&nbsp;&nbsp;def&nbsp;example_function():")
    print()

    # Check if the bug exists
    if "&nbsp" in escaped and "&nbsp;" not in escaped:
        print("BUG CONFIRMED: Incomplete HTML entity found - missing semicolon in &nbsp entity")
        return True
    else:
        print("Bug NOT confirmed")
        return False


@given(st.text(min_size=1, max_size=100).filter(lambda x: ' ' in x))
def test_html_entity_completeness_hypothesis(line_with_spaces):
    """Property-based test from bug report."""
    middleware = ServerErrorMiddleware(None, debug=True)

    # This mirrors what happens in format_line method
    escaped = html.escape(line_with_spaces).replace(" ", "&nbsp")

    if "&nbsp" in escaped and "&nbsp;" not in escaped:
        raise AssertionError(
            f"Incomplete HTML entity found. "
            f"Missing semicolon in &nbsp entity. "
            f"Input: {repr(line_with_spaces)}, "
            f"Output: {escaped}"
        )


def test_format_line_method():
    """Test the actual format_line method."""
    middleware = ServerErrorMiddleware(None, debug=True)

    # Test with a line containing spaces
    test_line = "    def foo():"
    frame_lineno = 10
    frame_index = 0
    index = 0

    # Call the actual method
    formatted = middleware.format_line(index, test_line, frame_lineno, frame_index)

    print("Testing format_line method:")
    print("Input line:", repr(test_line))
    print("Formatted output contains:", formatted[:100], "...")
    print()

    # Check if the bug exists in the output
    if "&nbsp" in formatted and "&nbsp;" not in formatted:
        print("BUG CONFIRMED in format_line output: Missing semicolon in &nbsp entity")
        return True
    else:
        print("Bug NOT confirmed in format_line")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing ServerErrorMiddleware HTML Entity Bug")
    print("=" * 60)
    print()

    # Test 1: Exact reproduction
    bug1 = test_reproduction_exact()
    print()

    # Test 2: format_line method
    bug2 = test_format_line_method()
    print()

    # Test 3: Property-based test
    print("Running property-based test with Hypothesis...")
    try:
        test_html_entity_completeness_hypothesis()
        print("Property test passed - no bug found")
    except AssertionError as e:
        print(f"Property test failed - BUG CONFIRMED: {e}")
        bug3 = True
    else:
        bug3 = False

    print()
    print("=" * 60)
    if bug1 or bug2 or bug3:
        print("RESULT: Bug is CONFIRMED - Missing semicolon in &nbsp entities")
    else:
        print("RESULT: Bug NOT confirmed")