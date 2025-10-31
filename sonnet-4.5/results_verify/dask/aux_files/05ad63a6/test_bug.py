#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st, settings
import time

print("Testing the property-based test that claims to find a bug...")

@given(
    st.integers(min_value=1, max_value=10**9),
    st.integers(min_value=1, max_value=10**8)
)
@settings(max_examples=10)  # Reduced for quick testing
def test_blocksize_calculation_terminates_quickly(size, blocksize):
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    iterations = 0
    start = time.time()

    while size - place > (blocksize1 * 2) - 1:
        iterations += 1
        place += blocksize1

        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"SLOW: size={size}, blocksize={blocksize}, iterations={iterations}, elapsed={elapsed:.2f}s")
            raise AssertionError(
                f"Loop did not terminate in reasonable time. "
                f"size={size}, blocksize={blocksize}, "
                f"iterations={iterations}, elapsed={elapsed:.2f}s"
            )

    print(f"OK: size={size}, blocksize={blocksize}, iterations={iterations}")

# Test with the specific failing input mentioned
print("\nTesting specific failing input: size=141335, blocksize=1")
size = 141335
blocksize = 1

if size % blocksize and size > blocksize:
    blocksize1 = size / (size // blocksize)
else:
    blocksize1 = blocksize

place = 0
iterations = 0
start = time.time()

while size - place > (blocksize1 * 2) - 1:
    iterations += 1
    place += blocksize1

elapsed = time.time() - start
print(f"Result: iterations={iterations}, elapsed={elapsed:.3f}s")

# Now test the extreme case
print("\nTesting extreme case: size=1,000,000,000, blocksize=1")
size = 1_000_000_000
blocksize = 1

if size % blocksize and size > blocksize:
    blocksize1 = size / (size // blocksize)
else:
    blocksize1 = blocksize

place = 0
iterations = 0
start = time.time()

while size - place > (blocksize1 * 2) - 1:
    iterations += 1
    place += blocksize1

    if iterations == 1_000_000:
        elapsed = time.time() - start
        print(f"Already at {iterations:,} iterations after {elapsed:.3f}s - this will take ~500 million total")
        estimated_total_time = elapsed * (size // (blocksize * 2)) / iterations
        print(f"Estimated total time: {estimated_total_time:.1f} seconds")
        break

print(f"Expected total iterations: ~{size // (blocksize * 2):,}")

# Run the hypothesis test
print("\n\nRunning hypothesis test...")
try:
    test_blocksize_calculation_terminates_quickly()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")