#!/usr/bin/env python3
"""Test with Hypothesis as reported in the bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import django.template

@given(st.integers(min_value=0, max_value=1000))
def test_integer_trailing_period_property(num):
    text = f"{num}."
    var = django.template.Variable(text)

    if var.literal is not None and var.lookups is not None:
        print(f"FAILED for num={num}: Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'")
        assert False, f"Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'"
    else:
        print(f"OK for num={num}")

# Run the test
print("Running Hypothesis test...")
try:
    test_integer_trailing_period_property()
except AssertionError as e:
    print(f"AssertionError caught: {e}")
    print("\nThe hypothesis test FAILS as reported")