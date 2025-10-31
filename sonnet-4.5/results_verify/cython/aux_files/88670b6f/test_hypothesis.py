#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@settings(max_examples=500)
@given(st.text())
def test_parse_list_returns_list(s):
    try:
        result = parse_list(s)
        assert isinstance(result, list), f"parse_list should return a list, got {type(result)}"
        print(f"✓ Input '{s[:20]}{'...' if len(s) > 20 else ''}' returned list: {result[:3] if len(result) > 3 else result}")
    except Exception as e:
        print(f"✗ Input '{s[:20]}{'...' if len(s) > 20 else ''}' raised {type(e).__name__}: {e}")
        raise

# Run the test
test_parse_list_returns_list()