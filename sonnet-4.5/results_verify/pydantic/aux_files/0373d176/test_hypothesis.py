#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from pydantic.alias_generators import to_snake

@given(st.text(min_size=1))
@settings(max_examples=1000)
@example('A0')  # Add the failing example
def test_to_snake_idempotent(field_name):
    """to_snake applied twice should equal to_snake applied once (idempotence)."""
    once = to_snake(field_name)
    twice = to_snake(once)
    assert once == twice, f"Failed for input '{field_name}': to_snake('{field_name}') = '{once}', to_snake('{once}') = '{twice}'"

if __name__ == "__main__":
    try:
        test_to_snake_idempotent()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")