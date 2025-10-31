#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import pytest
from hypothesis import given, strategies as st
from django.db.backends.dummy.base import DatabaseWrapper


@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.text())))
def test_operators_identity_differs_between_instances(settings):
    wrapper1 = DatabaseWrapper(settings)
    wrapper2 = DatabaseWrapper(settings)

    assert (
        wrapper1.operators is not wrapper2.operators
    ), "operators dict is the same object for different instances"

# Run the test
if __name__ == "__main__":
    # Run the raw test without hypothesis decorator
    def raw_test(settings):
        wrapper1 = DatabaseWrapper(settings)
        wrapper2 = DatabaseWrapper(settings)

        assert (
            wrapper1.operators is not wrapper2.operators
        ), "operators dict is the same object for different instances"

    # Run with a simple test case
    try:
        raw_test({})
        print("Test PASSED with empty dict")
    except AssertionError as e:
        print(f"Test FAILED with empty dict: {e}")

    # Run with another test case
    try:
        raw_test({'key': 'value'})
        print("Test PASSED with non-empty dict")
    except AssertionError as e:
        print(f"Test FAILED with non-empty dict: {e}")