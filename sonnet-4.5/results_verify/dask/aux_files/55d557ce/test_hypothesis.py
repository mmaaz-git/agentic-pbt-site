#!/usr/bin/env python3
"""Property-based test from the bug report"""

from hypothesis import given, strategies as st, settings
from dask.array.overlap import _overlap_internal_chunks


@given(
    num_dims=st.integers(min_value=1, max_value=4),
    depth=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=100)  # Reduced from 500 for faster testing
def test_overlap_internal_chunks_inconsistent_types(num_dims, depth):
    chunks = []
    for i in range(num_dims):
        if i % 2 == 0:
            chunks.append(tuple([10]))
        else:
            chunks.append(tuple([5, 5]))

    chunks = tuple(chunks)
    axes = {i: depth for i in range(num_dims)}

    result = _overlap_internal_chunks(chunks, axes)

    types_are_consistent = all(isinstance(r, tuple) for r in result) or \
                          all(isinstance(r, list) for r in result)

    if not types_are_consistent:
        print(f"\nFailed on input:")
        print(f"  chunks={chunks}")
        print(f"  axes={axes}")
        print(f"  result={result}")
        print(f"  result types={[type(r).__name__ for r in result]}")

    assert types_are_consistent, \
        f"Type inconsistency: result contains {[type(r).__name__ for r in result]}"


if __name__ == "__main__":
    # Run the test
    print("Running property-based test...")
    try:
        test_overlap_internal_chunks_inconsistent_types()
        print("Test completed - no failures found")
    except AssertionError as e:
        print(f"Test failed: {e}")