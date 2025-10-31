#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

from hypothesis import given, strategies as st, settings
from dask.array.overlap import ensure_minimum_chunksize

# First, let's run the property-based test from the bug report
@settings(max_examples=200)
@given(
    chunks=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=10),
    size=st.integers(min_value=1, max_value=30)
)
def test_ensure_minimum_chunksize_property_all_chunks_at_least_size(chunks, size):
    chunks = tuple(chunks)
    if sum(chunks) < size:
        return

    try:
        result = ensure_minimum_chunksize(size, chunks)

        for chunk in result:
            assert chunk >= size, f"Chunk {chunk} is less than minimum size {size}"
    except ValueError as e:
        if "larger than your array" not in str(e):
            raise

# Run the property test
print("Running property-based test...")
test_ensure_minimum_chunksize_property_all_chunks_at_least_size()
print("Property test passed!")

# Now reproduce the specific example from the bug report
print("\nReproducing the specific example:")
result = ensure_minimum_chunksize(10, (20, 20, 1))
print(f"ensure_minimum_chunksize(10, (20, 20, 1)) = {result}")
print(f"All chunks >= 10: {all(c >= 10 for c in result)}")

# Let's test a few more examples to understand the behavior
print("\nAdditional examples:")
test_cases = [
    (5, (10, 10, 10)),
    (5, (3, 3, 3)),
    (10, (5, 5, 5)),
    (7, (2, 2, 2, 2, 2)),
]

for size, chunks in test_cases:
    result = ensure_minimum_chunksize(size, chunks)
    print(f"ensure_minimum_chunksize({size}, {chunks}) = {result}")
    print(f"  All chunks >= {size}: {all(c >= size for c in result)}")
    print(f"  Sum preserved: {sum(chunks)} -> {sum(result)}")