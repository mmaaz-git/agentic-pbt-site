#!/usr/bin/env python3
"""Test script to reproduce the is_re_compilable bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.api.types import is_re_compilable
import re

print("Testing is_re_compilable with invalid regex patterns:")
print("-" * 50)

# Test cases from the bug report
test_cases = ['[', ')', '?', '*', '(', '{', '+', '^$*']

for pattern in test_cases:
    print(f"\nTesting: '{pattern}'")
    try:
        result = is_re_compilable(pattern)
        print(f"  is_re_compilable('{pattern}') = {result}")
    except Exception as e:
        print(f"  is_re_compilable('{pattern}') raised: {type(e).__name__}: {e}")

    # Compare with what re.compile does
    try:
        re.compile(pattern)
        print(f"  re.compile('{pattern}') succeeded")
    except TypeError as e:
        print(f"  re.compile('{pattern}') raised TypeError: {e}")
    except re.error as e:
        print(f"  re.compile('{pattern}') raised re.error: {e}")

print("\n" + "=" * 50)
print("Testing valid patterns and non-string objects:")
print("-" * 50)

# Test valid patterns and non-string objects
other_tests = ['.*', 'foo', r'\d+', 1, None, [], {}]

for obj in other_tests:
    print(f"\nTesting: {repr(obj)}")
    try:
        result = is_re_compilable(obj)
        print(f"  is_re_compilable({repr(obj)}) = {result}")
    except Exception as e:
        print(f"  is_re_compilable({repr(obj)}) raised: {type(e).__name__}: {e}")

    # Compare with what re.compile does
    try:
        re.compile(obj)
        print(f"  re.compile({repr(obj)}) succeeded")
    except TypeError as e:
        print(f"  re.compile({repr(obj)}) raised TypeError: {e}")
    except re.error as e:
        print(f"  re.compile({repr(obj)}) raised re.error: {e}")