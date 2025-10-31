#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Test 1: Hypothesis test
from hypothesis import given, strategies as st
import pytest
from django.db.models.functions import Substr, Left
from django.db.models.expressions import Value

print("Test 1: Manual property test (simulating hypothesis)")
def test_property_substr_should_reject_non_positive_length(length):
    """Property: Substr should reject non-positive length values, like Left does."""
    print(f"Testing Substr with length={length}")
    try:
        result = Substr(Value("test"), 1, length)
        print(f"  Substr succeeded with length={length} - No exception raised")
        return False  # Should have raised an exception
    except ValueError as e:
        print(f"  Substr raised ValueError: {e}")
        return True  # Expected behavior

# Run a few test cases
test_values = [0, -1, -5, -100]
for val in test_values:
    test_property_substr_should_reject_non_positive_length(val)

print("\n" + "="*60)
print("Test 2: Direct comparison of Left vs Substr behavior")
print("="*60)

# Test Left function with invalid lengths
print("\nTesting Left function:")
for length in [0, -5]:
    try:
        result = Left(Value("hello"), length)
        print(f"  Left(Value('hello'), {length}) - No exception raised")
    except ValueError as e:
        print(f"  Left(Value('hello'), {length}) - Raised ValueError: {e}")

# Test Substr function with invalid lengths
print("\nTesting Substr function:")
for length in [0, -5]:
    try:
        result = Substr(Value("hello"), 1, length)
        print(f"  Substr(Value('hello'), 1, {length}) - No exception raised")
    except ValueError as e:
        print(f"  Substr(Value('hello'), 1, {length}) - Raised ValueError: {e}")

print("\n" + "="*60)
print("Test 3: Check other similar functions")
print("="*60)

# Check LPad and Repeat as mentioned in the bug report
from django.db.models.functions import LPad, Repeat

print("\nTesting LPad function:")
try:
    result = LPad(Value("test"), 0)
    print(f"  LPad with length=0 - No exception raised")
except (ValueError, TypeError) as e:
    print(f"  LPad with length=0 - Raised {type(e).__name__}: {e}")

try:
    result = LPad(Value("test"), -5)
    print(f"  LPad with length=-5 - No exception raised")
except (ValueError, TypeError) as e:
    print(f"  LPad with length=-5 - Raised {type(e).__name__}: {e}")

print("\nTesting Repeat function:")
try:
    result = Repeat(Value("test"), 0)
    print(f"  Repeat with times=0 - No exception raised")
except (ValueError, TypeError) as e:
    print(f"  Repeat with times=0 - Raised {type(e).__name__}: {e}")

try:
    result = Repeat(Value("test"), -5)
    print(f"  Repeat with times=-5 - No exception raised")
except (ValueError, TypeError) as e:
    print(f"  Repeat with times=-5 - Raised {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Summary: Comparing validation behavior across functions")
print("="*60)