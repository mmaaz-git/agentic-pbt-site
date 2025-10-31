#!/usr/bin/env python3
"""Reproduce the ProgressBar division by zero bug"""

from hypothesis import given, strategies as st
from django.core.serializers.base import ProgressBar
from io import StringIO

print("Testing ProgressBar with Hypothesis property-based test...")

# First, run the hypothesis test
@given(st.integers(min_value=1, max_value=100))
def test_progressbar_handles_zero_total_count(count):
    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(count)

# Run the hypothesis test
try:
    test_progressbar_handles_zero_total_count()
    print("Hypothesis test passed (should have failed)")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Now run the simple reproducer
print("Testing with simple reproducer...")
try:
    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(1)
    print("Simple test passed (should have failed)")
except ZeroDivisionError as e:
    print(f"Simple test failed with ZeroDivisionError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Let's also test where the ProgressBar would be used in real code
print("Testing in context of serializer usage (default object_count=0)...")
from django.core.serializers import serialize

# Create a dummy queryset (empty list in this case)
queryset = []

try:
    # This should replicate the usage in the serialize method
    output = StringIO()
    pb = ProgressBar(output, 0)  # object_count defaults to 0
    for count, obj in enumerate(queryset, start=1):
        pb.update(count)  # This will never be called for empty queryset
    print("Empty queryset doesn't trigger the bug (no update calls)")

    # But what if we manually call update?
    pb.update(1)
    print("Manual update call passed (should have failed)")
except ZeroDivisionError as e:
    print(f"Manual update call failed with ZeroDivisionError: {e}")