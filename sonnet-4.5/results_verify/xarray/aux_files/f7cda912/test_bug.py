#!/usr/bin/env python3
"""Reproduce the bug reported for build_grid_chunks"""

# First, test the hypothesis property test
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.backends.chunks import build_grid_chunks

print("Testing with Hypothesis...")
@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_sum(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)

    assert sum(chunks) == size, \
        f"Sum of chunks {sum(chunks)} != size {size}"

    assert all(c > 0 for c in chunks), \
        f"All chunks should be positive, got {chunks}"

# Run hypothesis test
try:
    test_build_grid_chunks_sum()
    print("Hypothesis test passed (shouldn't happen if bug exists)")
except AssertionError as e:
    print(f"Hypothesis test failed as expected: {e}")

print("\n" + "="*50 + "\n")

# Now run the specific failing example
print("Testing specific failing case: size=1, chunk_size=2")
result = build_grid_chunks(size=1, chunk_size=2)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected: 1")

try:
    assert sum(result) == 1
    print("Assertion passed (shouldn't happen if bug exists)")
except AssertionError:
    print("AssertionError: Sum doesn't match expected value")

print("\n" + "="*50 + "\n")

# Let's trace through the logic for this case
print("Tracing through the logic for size=1, chunk_size=2:")
size = 1
chunk_size = 2
region = None

if region is None:
    region = slice(0, size)
    print(f"  region = slice(0, {size}) = {region}")

region_start = region.start or 0
print(f"  region_start = {region_start}")

# First chunk calculation
first_chunk = chunk_size - (region_start % chunk_size)
print(f"  first_chunk = {chunk_size} - ({region_start} % {chunk_size}) = {first_chunk}")

chunks_on_region = [first_chunk]
print(f"  chunks_on_region after first chunk: {chunks_on_region}")

# Calculate how many full chunks to add
full_chunks_count = (size - chunks_on_region[0]) // chunk_size
print(f"  full_chunks_count = ({size} - {chunks_on_region[0]}) // {chunk_size} = {full_chunks_count}")

chunks_on_region.extend([chunk_size] * full_chunks_count)
print(f"  chunks_on_region after full chunks: {chunks_on_region}")

# Check if we need a final partial chunk
remaining = (size - chunks_on_region[0]) % chunk_size
print(f"  remaining = ({size} - {chunks_on_region[0]}) % {chunk_size} = {remaining}")

if remaining != 0:
    chunks_on_region.append(remaining)
    print(f"  Added remaining chunk: {remaining}")

print(f"  Final chunks_on_region: {chunks_on_region}")
print(f"  Sum: {sum(chunks_on_region)}")

print("\n" + "="*50 + "\n")

# Let's test a few more edge cases
print("Testing other edge cases:")
test_cases = [
    (1, 1),  # size == chunk_size
    (1, 3),  # size < chunk_size
    (2, 3),  # size < chunk_size
    (3, 2),  # size > chunk_size
    (5, 2),  # size > chunk_size with remainder
    (10, 10), # size == chunk_size (larger)
    (10, 20), # size < chunk_size (larger)
]

for size, chunk_size in test_cases:
    result = build_grid_chunks(size, chunk_size)
    sum_result = sum(result)
    print(f"  size={size}, chunk_size={chunk_size}: chunks={result}, sum={sum_result}, expected={size}, OK={sum_result==size}")