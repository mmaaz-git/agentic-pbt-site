#!/usr/bin/env python3
"""Test the nbsp entity bug in Starlette"""

from hypothesis import given, strategies as st
import html

# Property-based test from the bug report
@given(st.text(min_size=1, max_size=100).filter(lambda x: " " in x))
def test_nbsp_entity_correctness(line):
    result = html.escape(line).replace(" ", "&nbsp")

    assert "&nbsp;" in result or " " not in line, (
        "HTML entity for non-breaking space should be '&nbsp;' with semicolon"
    )

# Simple reproduction test
def test_simple_reproduction():
    line = "    def example():"
    result = html.escape(line).replace(" ", "&nbsp")

    print(f"Result: {result}")
    print(f"Expected: {html.escape(line).replace(' ', '&nbsp;')}")
    print(f"\nInvalid entity: &nbsp (missing semicolon)")
    print(f"Valid entity: &nbsp; (with semicolon)")

    # Check if the bug exists
    assert "&nbsp;" not in result, "Bug confirmed: Missing semicolon in &nbsp entity"
    print("\nBug confirmed: The code produces &nbsp without semicolon")

if __name__ == "__main__":
    print("=== Running simple reproduction test ===")
    test_simple_reproduction()

    print("\n=== Running property-based test ===")
    try:
        test_nbsp_entity_correctness()
    except AssertionError as e:
        print(f"Property test failed as expected: {e}")
        print("This confirms the bug exists")