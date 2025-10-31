#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pandas.api import types
import re

failures = []

@given(st.one_of(st.text(), st.binary()))
@settings(max_examples=100)
def test_is_re_compilable_matches_re_compile(obj):
    global failures

    # What is_re_compilable returns
    try:
        result = types.is_re_compilable(obj)
        func_crashed = False
    except Exception as e:
        func_crashed = True
        result = None
        failures.append((obj, f"Function crashed: {type(e).__name__}: {e}"))

    # What re.compile actually does
    try:
        re.compile(obj)
        actually_compilable = True
    except (TypeError, re.error):
        actually_compilable = False

    # Check if behavior matches
    if not func_crashed:
        if result != actually_compilable:
            failures.append((obj, f"Mismatch: is_re_compilable={result}, actually_compilable={actually_compilable}"))
            assert False, f"Mismatch for {repr(obj)}: is_re_compilable={result}, actually_compilable={actually_compilable}"

# Run the test
try:
    test_is_re_compilable_matches_re_compile()
    print("Test completed")
except Exception as e:
    print(f"Test failed: {e}")

if failures:
    print(f"\nFound {len(failures)} failures:")
    for i, (obj, msg) in enumerate(failures[:10], 1):  # Show first 10
        print(f"{i}. Object: {repr(obj)[:50]}")
        print(f"   Error: {msg}")
else:
    print("\nNo failures found in hypothesis testing")