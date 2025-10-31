"""
Test script to reproduce the bug in xarray.backends.chunks.build_grid_chunks
"""
from hypothesis import given, strategies as st, settings
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.backends.chunks import build_grid_chunks

# First, let's run the hypothesis test
print("=== Running Hypothesis Test ===")

@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=1000)
def test_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size, region=None)
    actual_sum = sum(chunks)
    assert actual_sum == size, f"size={size}, chunk_size={chunk_size}, chunks={chunks}, sum={actual_sum}"

try:
    test_sum_equals_size()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now, let's reproduce the specific failure case
print("\n=== Reproducing Specific Bug Case ===")
result = build_grid_chunks(size=1, chunk_size=2, region=None)

print(f"Chunks: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected size: 1")
print(f"Test passes: {sum(result) == 1}")

# Let's test a few more edge cases
print("\n=== Testing More Edge Cases ===")
test_cases = [
    (1, 2),   # size < chunk_size
    (1, 10),  # size much smaller than chunk_size
    (2, 3),   # size < chunk_size
    (3, 5),   # size < chunk_size
    (10, 3),  # size > chunk_size (should work)
    (10, 10), # size == chunk_size
    (10, 20), # size < chunk_size
]

for size, chunk_size in test_cases:
    chunks = build_grid_chunks(size, chunk_size, region=None)
    actual_sum = sum(chunks)
    passes = actual_sum == size
    print(f"size={size:3}, chunk_size={chunk_size:3}, chunks={str(chunks):20}, sum={actual_sum:3}, passes={passes}")