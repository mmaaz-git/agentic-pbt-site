#!/usr/bin/env python3
"""Test script to reproduce the ProgressBar ZeroDivisionError bug."""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Test 1: Run the hypothesis test
print("Test 1: Running hypothesis property-based test...")
try:
    from hypothesis import given, strategies as st, assume
    from django.core.serializers.base import ProgressBar
    from io import StringIO

    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_progress_bar_no_crash(total_count, count):
        assume(count <= total_count)
        output = StringIO()
        pb = ProgressBar(output, total_count)
        pb.update(count)

    # Run the test
    test_progress_bar_no_crash()
    print("Hypothesis test completed - should have failed but didn't reach failing case")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")

print("\nTest 2: Direct reproduction with total_count=0...")
try:
    from io import StringIO
    from django.core.serializers.base import ProgressBar

    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(0)
    print("No error - unexpected!")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred as expected: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Different error occurred: {type(e).__name__}: {e}")

print("\nTest 3: Checking if the error occurs at line 59...")
try:
    from io import StringIO
    from django.core.serializers.base import ProgressBar
    import traceback

    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(0)
except ZeroDivisionError:
    tb = traceback.format_exc()
    if 'line 59' in tb:
        print("Confirmed: ZeroDivisionError occurs at line 59 in base.py")
    else:
        print("Error occurred but not at expected line")
    print("\nFull traceback:")
    print(tb)