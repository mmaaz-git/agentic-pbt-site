#!/usr/bin/env python3
"""Test script to reproduce the fix_bounds bug"""

from hypothesis import given, strategies as st, settings
from dask.diagnostics.profile_visualize import fix_bounds

# First, let's look at the actual implementation
import inspect
print("=== fix_bounds implementation ===")
print(inspect.getsource(fix_bounds))
print()

# Test with the specific failing input
print("=== Testing with specific failing input ===")
start = 6442450945.0
end = 0.0
min_span = 2147483647.9201343

new_start, new_end = fix_bounds(start, end, min_span)

actual_span = new_end - new_start
print(f"Input: start={start}, end={end}, min_span={min_span}")
print(f"Output: new_start={new_start}, new_end={new_end}")
print(f"Actual span: {actual_span}")
print(f"Expected min_span: {min_span}")
print(f"Difference: {min_span - actual_span}")
print(f"Bug (actual_span < min_span): {actual_span < min_span}")
print()

# Let's understand the floating point issue better
print("=== Understanding the floating point issue ===")
sum_result = start + min_span
print(f"start + min_span = {start} + {min_span} = {sum_result}")
print(f"Expected sum: {6442450945.0 + 2147483647.9201343}")
print(f"Actual sum: {sum_result}")
print(f"Lost precision: {(6442450945.0 + 2147483647.9201343) - sum_result}")
print()

# Now run the property-based test
print("=== Running property-based test ===")

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    end=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    min_span=st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e10)
)
@settings(max_examples=100)
def test_fix_bounds_span_invariant(start, end, min_span):
    new_start, new_end = fix_bounds(start, end, min_span)
    assert new_start == start, f"Start changed: {start} -> {new_start}"
    assert new_end - new_start >= min_span, f"Span too small: {new_end - new_start} < {min_span}"

try:
    test_fix_bounds_span_invariant()
    print("Property-based test passed with 100 examples")
except AssertionError as e:
    print(f"Property-based test failed: {e}")
except Exception as e:
    print(f"Property-based test error: {e}")