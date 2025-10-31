#!/usr/bin/env python3
"""Test script to reproduce the bug reported in pandas StringMethods.count()"""

import pandas as pd
import re

print("=" * 60)
print("Testing pandas StringMethods.count() with regex metacharacters")
print("=" * 60)

# Create test series
s = pd.Series(['test)test', 'hello(world', 'dot.here'])
print(f"Test Series: {s.tolist()}")
print()

# Test 1: Count closing parenthesis
print("Test 1: Counting ')' character")
try:
    result = s.str.count(')')
    print(f"  s.str.count(')'): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 2: Count opening parenthesis
print("\nTest 2: Counting '(' character")
try:
    result = s.str.count('(')
    print(f"  s.str.count('('): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 3: Count dot
print("\nTest 3: Counting '.' character")
try:
    result = s.str.count('.')
    print(f"  s.str.count('.'): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 4: Try with regex=False (should fail if parameter doesn't exist)
print("\nTest 4: Try s.str.count(')', regex=False)")
try:
    result = s.str.count(')', regex=False)
    print(f"  s.str.count(')', regex=False): {result.tolist()}")
except TypeError as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Expected results using Python's str.count()
print("\n" + "=" * 60)
print("Expected results using Python's str.count():")
print("=" * 60)
print(f"'test)test'.count(')'): {s.iloc[0].count(')')}")
print(f"'hello(world'.count('('): {s.iloc[1].count('(')}")
print(f"'dot.here'.count('.'): {s.iloc[2].count('.')}")

# Test with escaped characters
print("\n" + "=" * 60)
print("Workaround: Using re.escape() for literal matching")
print("=" * 60)
print("Test with re.escape():")
try:
    result = s.str.count(re.escape(')'))
    print(f"  s.str.count(re.escape(')')): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

try:
    result = s.str.count(re.escape('('))
    print(f"  s.str.count(re.escape('(')): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

try:
    result = s.str.count(re.escape('.'))
    print(f"  s.str.count(re.escape('.')): {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")