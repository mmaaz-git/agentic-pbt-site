#!/usr/bin/env python3
"""Test the reported ProgressBar division by zero bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from io import StringIO
from django.core.serializers.base import ProgressBar

# Test 1: Basic reproduction - this should cause ZeroDivisionError
print("Test 1: Basic reproduction with total_count=0")
try:
    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(1)  # This should crash with ZeroDivisionError
    print("  ERROR: No exception raised!")
except ZeroDivisionError as e:
    print(f"  ZeroDivisionError raised as expected: {e}")
except Exception as e:
    print(f"  Unexpected exception: {type(e).__name__}: {e}")

# Test 2: Check with different count values
print("\nTest 2: Different count values with total_count=0")
for count in [1, 10, 100]:
    try:
        output = StringIO()
        pb = ProgressBar(output, total_count=0)
        pb.update(count)
        print(f"  count={count}: No exception raised!")
    except ZeroDivisionError:
        print(f"  count={count}: ZeroDivisionError raised")

# Test 3: Normal operation with total_count > 0
print("\nTest 3: Normal operation with total_count > 0")
try:
    output = StringIO()
    pb = ProgressBar(output, total_count=100)
    pb.update(50)
    print(f"  Success! Output: {output.getvalue()!r}")
except Exception as e:
    print(f"  Unexpected exception: {type(e).__name__}: {e}")

# Test 4: Property-based test from bug report
print("\nTest 4: Property-based test from bug report")
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_progressbar_handles_zero_total_count_gracefully(count):
    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(count)  # This should crash

try:
    test_progressbar_handles_zero_total_count_gracefully()
    print("  ERROR: Property-based test passed when it should fail!")
except ZeroDivisionError:
    print("  Property-based test failed as expected with ZeroDivisionError")