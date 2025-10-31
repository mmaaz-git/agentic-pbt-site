#!/usr/bin/env python3
"""Hypothesis test for Cython Shadow index_type bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Shadow import index_type
import pytest


def test_index_type_tuple_with_multiple_steps_fails():
    """Step should only be provided once, not in multiple dimensions"""
    slices = (slice(None, None, 1), slice(None, None, 1))
    with pytest.raises(Exception):
        index_type(int, slices)


if __name__ == '__main__':
    # Run the test directly
    print("Running Hypothesis test for Cython Shadow index_type bug")
    print("=" * 60)

    print("\nTest: Step should only be provided once, not in multiple dimensions")
    print("Input: slices = (slice(None, None, 1), slice(None, None, 1))")

    # Try to run the test
    slices = (slice(None, None, 1), slice(None, None, 1))
    try:
        result = index_type(int, slices)
        print(f"\nFAILURE: Expected exception but got result: {result}")
        print("\nThe function should have raised an InvalidTypeSpecification exception")
        print("with message: 'Step may only be provided once, and only in the'")
        print("             'first or last dimension.'")
        print("\nThis is a bug in the step validation logic!")
    except Exception as e:
        print(f"\nSUCCESS: Raised exception as expected: {e}")