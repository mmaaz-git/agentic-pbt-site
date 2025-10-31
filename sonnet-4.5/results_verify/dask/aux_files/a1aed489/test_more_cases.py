#!/usr/bin/env python3
"""Find more cases where variance is computed incorrectly"""

from dask.bag.chunk import var_chunk, var_aggregate
import random

print("Testing various cases to find incorrect variance calculations:\n")

# Test 1: All identical large positive values
def test_identical_values(value, count=3):
    values = [value] * count
    chunk_result = var_chunk(values)
    variance = var_aggregate([chunk_result], ddof=0)
    expected = 0.0

    if abs(variance) > 1e-10:  # Should be exactly 0
        print(f"FAIL: Identical values {value} (x{count})")
        print(f"  Got variance: {variance}, expected: {expected}")
        print(f"  Error: {abs(variance - expected)}")
        return False
    return True

# Test 2: Values with known variance
def test_known_variance():
    # Simple case: [0, 1] has variance 0.25
    values = [0, 1]
    chunk_result = var_chunk(values)
    variance = var_aggregate([chunk_result], ddof=0)
    expected = 0.25

    if abs(variance - expected) > 1e-10:
        print(f"FAIL: Known variance test [0, 1]")
        print(f"  Got variance: {variance}, expected: {expected}")
        return False

    # Another simple case: [-1, 0, 1] has variance 2/3
    values = [-1, 0, 1]
    chunk_result = var_chunk(values)
    variance = var_aggregate([chunk_result], ddof=0)
    expected = 2/3

    if abs(variance - expected) > 1e-10:
        print(f"FAIL: Known variance test [-1, 0, 1]")
        print(f"  Got variance: {variance}, expected: {expected}")
        return False

    return True

# Test identical values at different scales
print("Testing identical values at different scales:")
test_values = [1e-10, 1e-5, 1, 100, 1e5, 1e8, 1e10, -1e5, -1e8]
for val in test_values:
    if not test_identical_values(val):
        print()

print("\nTesting known variance cases:")
test_known_variance()

# Test cases where numerical instability occurs
print("\nTesting numerically challenging cases:")

# Large values with small differences
values = [1e8, 1e8 + 1, 1e8 - 1]
chunk_result = var_chunk(values)
variance = var_aggregate([chunk_result], ddof=0)
mean = sum(values) / len(values)
expected = sum((x - mean) ** 2 for x in values) / len(values)
print(f"Large values with small differences: {values}")
print(f"  Computed variance: {variance}")
print(f"  Expected variance: {expected}")
print(f"  Relative error: {abs(variance - expected) / expected if expected != 0 else 'N/A'}")
print()

# Very large identical values
values = [1e15] * 10
chunk_result = var_chunk(values)
variance = var_aggregate([chunk_result], ddof=0)
print(f"Very large identical values: [1e15] * 10")
print(f"  Computed variance: {variance}")
print(f"  Expected variance: 0.0")
print(f"  Is negative? {variance < 0}")
print()

# Find threshold where identical values start giving wrong results
print("Finding threshold for numerical errors with identical values:")
for exp in range(1, 20):
    val = 10 ** exp
    values = [val, val, val]
    chunk_result = var_chunk(values)
    variance = var_aggregate([chunk_result], ddof=0)

    if abs(variance) > 1e-15:
        print(f"  10^{exp}: variance = {variance} (should be 0)")

print("\nTesting negative variance cases:")
# Generate random large values to find negative variance
random.seed(42)
negative_count = 0
for _ in range(100):
    # Generate identical or nearly identical large values
    base = random.uniform(-1e6, 1e6)
    values = [base] * random.randint(2, 5)

    chunk_result = var_chunk(values)
    variance = var_aggregate([chunk_result], ddof=0)

    if variance < 0:
        negative_count += 1
        if negative_count <= 5:  # Show first 5 cases
            print(f"  Negative variance: {variance} for values like {base}")