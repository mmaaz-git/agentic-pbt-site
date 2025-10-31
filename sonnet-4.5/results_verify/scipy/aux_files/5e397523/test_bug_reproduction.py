#!/usr/bin/env python3
"""Test to reproduce the build_grid_chunks bug reported."""

import sys
import os

# Add the xarray_env to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks
from hypothesis import given, settings, strategies as st

# Test the hypothesis test case
@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size, f"size={size}, chunk_size={chunk_size}, chunks={chunks}, sum={sum(chunks)}"

# Test specific failing case
print("Testing specific failing case: size=1, chunk_size=2")
result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected: 1")
print()

# Let's test a few more edge cases
test_cases = [
    (1, 1),   # Equal size and chunk_size
    (1, 2),   # chunk_size > size (the reported bug)
    (1, 10),  # Much larger chunk_size
    (2, 1),   # size > chunk_size
    (10, 3),  # Normal case
    (3, 10),  # chunk_size > size
    (5, 5),   # Equal
    (7, 3),   # Normal with remainder
]

print("Testing various cases:")
for size, chunk_size in test_cases:
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    chunks_sum = sum(chunks)
    status = "✓" if chunks_sum == size else "✗"
    print(f"{status} size={size:3}, chunk_size={chunk_size:3}, chunks={chunks}, sum={chunks_sum}")

print("\nRunning hypothesis test...")
try:
    test_build_grid_chunks_sum_invariant()
    print("Hypothesis test passed (no counterexamples found)!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")