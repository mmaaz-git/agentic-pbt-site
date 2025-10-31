#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify that _sqlite_format_dtdelta
always returns strings, regardless of the operation type.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta


@given(st.floats(min_value=0.1, max_value=1e6), st.floats(min_value=0.1, max_value=1e6))
@settings(max_examples=100)
def test_format_dtdelta_always_returns_string(lhs, rhs):
    """Test that _sqlite_format_dtdelta always returns a string for all operations."""
    for connector in ["+", "-", "*", "/"]:
        result = _sqlite_format_dtdelta(connector, lhs, rhs)
        if result is not None:
            assert isinstance(result, str), f"format_dtdelta({connector!r}, {lhs}, {rhs}) should return string, got {type(result)}"


if __name__ == "__main__":
    # Run the property-based test
    try:
        test_format_dtdelta_always_returns_string()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates the bug where multiplication (*) and division (/)")
        print("operations return float values instead of formatted strings.")