#!/usr/bin/env python3
"""Deeper analysis of the RangeIndex.arange behavior"""

from xarray.indexes import RangeIndex
import numpy as np
import math

def analyze_arange(start, stop, step):
    """Analyze how RangeIndex.arange processes the parameters"""
    print(f"\nInput: start={start}, stop={stop}, step={step}")

    # Simulate what arange does (line 219)
    size = math.ceil((stop - start) / step)
    print(f"  Calculated size: math.ceil(({stop} - {start}) / {step}) = {size}")

    # Create the index
    index = RangeIndex.arange(start, stop, step, dim="x")

    # What the transform stores
    print(f"  Transform.start: {index.transform.start}")
    print(f"  Transform.stop: {index.transform.stop}")
    print(f"  Transform.size: {index.transform.size}")
    print(f"  Transform._step: {index.transform._step}")

    # How the step is calculated (lines 58-65 in RangeCoordinateTransform)
    calculated_step = (index.transform.stop - index.transform.start) / index.transform.size if index.transform.size > 0 else 1.0
    print(f"  Calculated step: ({index.transform.stop} - {index.transform.start}) / {index.transform.size} = {calculated_step}")

    # Get the actual values
    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]
    print(f"  Generated values: {values}")

    # Compare with what numpy.arange would produce
    numpy_values = np.arange(start, stop, step)
    print(f"  numpy.arange values: {numpy_values}")

    # Check if they match
    values_match = len(values) == len(numpy_values) and np.allclose(values, numpy_values, rtol=1e-9)
    print(f"  Values match numpy.arange: {values_match}")

    return values_match

# Test various cases
test_cases = [
    (0.0, 1.5, 1.0),  # The reported failing case
    (0.0, 2.0, 1.0),  # A case that might work
    (0.0, 10.0, 3.0), # Another failing case
    (0.0, 1.0, 0.3),  # Non-exact division case
]

print("=" * 60)
print("Analysis of RangeIndex.arange vs numpy.arange behavior")
print("=" * 60)

for start, stop, step in test_cases:
    analyze_arange(start, stop, step)