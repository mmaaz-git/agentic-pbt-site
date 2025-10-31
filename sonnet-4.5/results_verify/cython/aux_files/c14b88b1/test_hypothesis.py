#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list
import pytest

# Run the hypothesis test to find failing inputs
@given(st.text())
@settings(max_examples=1000)
def test_parse_list_handles_all_inputs(s):
    try:
        result = parse_list(s)
        assert isinstance(result, list)
    except (KeyError, IndexError):
        print(f"Failed on input: {repr(s)}")
        pytest.fail(f"parse_list should handle input gracefully: {s!r}")

if __name__ == "__main__":
    # Run the test
    test_parse_list_handles_all_inputs()
    print("Hypothesis test completed")