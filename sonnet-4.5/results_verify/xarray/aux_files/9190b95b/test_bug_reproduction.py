#!/usr/bin/env python3

import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env')

# First, let's test the manual reproduction
print("=" * 60)
print("MANUAL REPRODUCTION TEST")
print("=" * 60)

import xarray.plot.utils as plot_utils

coord = np.array([0.0, 1.0, 0.0])
breaks = plot_utils._infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Inferred breaks:   {breaks}")
print(f"Breaks range: [{min(breaks)}, {max(breaks)}]")

# Check if all coordinates are within the breaks range
try:
    assert all(min(breaks) <= c <= max(breaks) for c in coord)
    print("✓ All coordinates are within the breaks range")
except AssertionError:
    print("✗ AssertionError: Not all coordinates are within the breaks range")
    for i, c in enumerate(coord):
        if not (min(breaks) <= c <= max(breaks)):
            print(f"  - coordinate[{i}] = {c} is NOT in range [{min(breaks)}, {max(breaks)}]")

# Now let's run the hypothesis test
print("\n" + "=" * 60)
print("HYPOTHESIS TEST")
print("=" * 60)

from hypothesis import given, strategies as st, settings
import traceback

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=2, max_size=20)
)
@settings(max_examples=100)
def test_infer_interval_breaks_contains_original(values):
    """Property: interval breaks should contain all original points"""
    arr = np.array(values)
    breaks = plot_utils._infer_interval_breaks(arr)

    for i in range(len(arr)):
        min_break = min(breaks)
        max_break = max(breaks)
        assert min_break <= arr[i] <= max_break, \
            f"Original value {arr[i]} not in range [{min_break}, {max_break}]"

try:
    test_infer_interval_breaks_contains_original()
    print("✓ Hypothesis test passed")
except Exception as e:
    print(f"✗ Hypothesis test failed: {e}")
    print("\nTraceback:")
    traceback.print_exc()

# Let's also test with the specific failing input
print("\n" + "=" * 60)
print("SPECIFIC FAILING INPUT TEST")
print("=" * 60)

failing_input = [0.0, 1.0, 0.0]
arr = np.array(failing_input)
breaks = plot_utils._infer_interval_breaks(arr)

print(f"Failing input: {failing_input}")
print(f"Array: {arr}")
print(f"Breaks: {breaks}")
print(f"Breaks range: [{min(breaks)}, {max(breaks)}]")

for i in range(len(arr)):
    min_break = min(breaks)
    max_break = max(breaks)
    in_range = min_break <= arr[i] <= max_break
    print(f"  arr[{i}] = {arr[i]}: {'✓' if in_range else '✗'} in range")