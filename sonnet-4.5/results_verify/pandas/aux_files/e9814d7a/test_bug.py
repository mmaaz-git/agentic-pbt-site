#!/usr/bin/env python3
"""Test the reported bug in pandas.core.dtypes.inference.is_re_compilable"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.dtypes.inference import is_re_compilable
import re

# Test 1: Simple invalid regex pattern from bug report
print("Test 1: Testing with ')'")
try:
    result = is_re_compilable(")")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 2: Another invalid pattern
print("\nTest 2: Testing with '?'")
try:
    result = is_re_compilable("?")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 3: Valid pattern
print("\nTest 3: Testing with '.*'")
try:
    result = is_re_compilable(".*")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 4: Non-string object
print("\nTest 4: Testing with integer 1")
try:
    result = is_re_compilable(1)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 5: None object
print("\nTest 5: Testing with None")
try:
    result = is_re_compilable(None)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 6: Check what re.compile actually raises for invalid patterns
print("\n--- Testing re.compile behavior directly ---")
print("Test re.compile(')')")
try:
    re.compile(")")
    print("  Success")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

print("\nTest re.compile('?')")
try:
    re.compile("?")
    print("  Success")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

print("\nTest re.compile(1)")
try:
    re.compile(1)
    print("  Success")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")