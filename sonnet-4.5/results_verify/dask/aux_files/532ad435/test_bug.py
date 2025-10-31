import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

# First, let's test the specific values mentioned in the bug report
test_values = [
    100 * 2**50,
    2**60 - 1,
]

print("Testing specific values from bug report:")
for value in test_values:
    result = format_bytes(value)
    length = len(result)
    print(f"format_bytes({value}) = '{result}' (length: {length})")
    if length > 10:
        print(f"  VIOLATION: Length {length} > 10 for value < 2**60")

# Let's test more values to understand the boundary
print("\nTesting additional values to find the boundary:")

# Test values around 100 * 2**50
for multiplier in [99, 99.9, 100, 101]:
    value = int(multiplier * 2**50)
    if value < 2**60:
        result = format_bytes(value)
        length = len(result)
        print(f"format_bytes({multiplier} * 2**50) = '{result}' (length: {length})")

# Test the hypothesis test
print("\nRunning property-based test with hypothesis:")
from hypothesis import given, strategies as st, settings

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

try:
    test_format_bytes_output_length()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Let's verify the math
print("\nMath verification:")
print(f"100 * 2**50 = {100 * 2**50}")
print(f"2**60 = {2**60}")
print(f"100 * 2**50 < 2**60: {100 * 2**50 < 2**60}")

# Check formatting of 100.00 PiB
print(f"\nFormatted: {100 * 2**50 / 2**50:.2f} PiB")
print(f"Length of '100.00 PiB': {len('100.00 PiB')}")

# Find exact boundary where length exceeds 10
print("\nFinding exact boundary where length > 10:")
for i in range(99, 101):
    for j in range(10):
        value = int((i + j/10) * 2**50)
        if value < 2**60:
            result = format_bytes(value)
            if len(result) > 10:
                print(f"First violation at {i + j/10} * 2**50: '{result}' (length: {len(result)})")