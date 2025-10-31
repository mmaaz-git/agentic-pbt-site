#!/usr/bin/env python3
"""Test the reported bug in RangeIndex.arange step parameter"""

# First, let's run the simple reproduction case from the bug report
from xarray.indexes import RangeIndex
import numpy as np

print("=== Simple Reproduction Test ===")
index = RangeIndex.arange(0.0, 1.5, 1.0, dim="x")

print(f"Expected step: 1.0")
print(f"Actual step: {index.step}")

coords = index.transform.forward({index.dim: np.arange(index.size)})
values = coords[index.coord_name]

print(f"Expected values: [0.0, 1.0]")
print(f"Actual values: {values}")
print()

# Now let's test the property-based test from the report
print("=== Property-Based Test ===")
from hypothesis import given, strategies as st, settings, assume

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50)  # Reduced for faster testing
def test_arange_step_matches_parameter(start, stop, step):
    assume(stop > start)
    assume(step > 0)
    assume((stop - start) / step < 1e6)

    index = RangeIndex.arange(start, stop, step, dim="x")

    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]

    if index.size > 1:
        actual_steps = np.diff(values)

        try:
            assert np.allclose(actual_steps, step, rtol=1e-9)
            return True
        except AssertionError:
            print(f"FAILED: start={start}, stop={stop}, step={step}")
            print(f"  Expected step: {step}")
            print(f"  Actual steps: {actual_steps}")
            print(f"  Index.step: {index.step}")
            return False
    return True

# Run the property test
print("Running property-based tests...")
try:
    test_arange_step_matches_parameter()
    print("Property test passed!")
except Exception as e:
    print(f"Property test failed with error: {e}")

# Test with some specific edge cases
print("\n=== Edge Cases ===")

test_cases = [
    (0.0, 1.5, 1.0),
    (0.0, 2.0, 1.0),
    (0.0, 10.0, 3.0),
    (1.0, 5.5, 2.0),
    (0.0, 1.0, 0.3),
]

for start, stop, step in test_cases:
    print(f"\nTest case: start={start}, stop={stop}, step={step}")
    index = RangeIndex.arange(start, stop, step, dim="x")

    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]

    print(f"  Size: {index.size}")
    print(f"  Expected step: {step}")
    print(f"  Index.step: {index.step}")
    print(f"  Values: {values}")

    if index.size > 1:
        actual_steps = np.diff(values)
        print(f"  Actual steps between values: {actual_steps}")
        is_correct = np.allclose(actual_steps, step, rtol=1e-9)
        print(f"  Steps match expected: {is_correct}")