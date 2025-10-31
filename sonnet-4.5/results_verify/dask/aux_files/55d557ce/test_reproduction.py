#!/usr/bin/env python3
"""Test to reproduce the type inconsistency bug in _overlap_internal_chunks"""

from dask.array.overlap import _overlap_internal_chunks

# Test case from the bug report
chunks = ((10,), (5, 5), (10,))
axes = {0: 0, 1: 0, 2: 0}

result = _overlap_internal_chunks(chunks, axes)

print(f"Input:  {chunks}")
print(f"Result: {result}")
print(f"Result element types: {[type(r).__name__ for r in result]}")

# Check assertions from the bug report
print(f"\nAssertion checks:")
print(f"result[0] is tuple: {isinstance(result[0], tuple)}")
print(f"result[1] is list: {isinstance(result[1], list)}")
print(f"result[2] is tuple: {isinstance(result[2], tuple)}")

# Test with different axes values to see if depth affects the behavior
print("\n--- Testing with non-zero depth ---")
axes_with_depth = {0: 1, 1: 2, 2: 1}
result_with_depth = _overlap_internal_chunks(chunks, axes_with_depth)
print(f"Input chunks: {chunks}")
print(f"Axes: {axes_with_depth}")
print(f"Result: {result_with_depth}")
print(f"Result element types: {[type(r).__name__ for r in result_with_depth]}")