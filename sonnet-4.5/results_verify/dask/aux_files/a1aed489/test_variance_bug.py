#!/usr/bin/env python3
"""Test to reproduce the variance bug report"""

from hypothesis import given, strategies as st, settings
import math
from dask.bag.chunk import var_chunk, var_aggregate

# First, let's test the specific failing input mentioned
print("Testing specific failing input:")
values = [-356335.16553451226, -356335.16553451226, -356335.16553451226]
chunk_result = var_chunk(values)
variance = var_aggregate([chunk_result], ddof=0)

print(f"Input values: {values}")
print(f"Chunk result: {chunk_result}")
print(f"Computed variance: {variance}")
print(f"Expected variance: 0.0")
print(f"Is variance negative? {variance < 0}")
print()

# Test another case
print("Testing second case:")
values2 = [259284.59765625, 259284.59765625, 259284.59765625]
chunk_result2 = var_chunk(values2)
variance2 = var_aggregate([chunk_result2], ddof=0)

print(f"Input values: {values2}")
print(f"Chunk result: {chunk_result2}")
print(f"Computed variance: {variance2}")
print(f"Expected variance: 0.0")
print(f"Is variance negative? {variance2 < 0}")
print()

# Let's manually verify the math
print("Manual verification of variance calculation:")
values3 = [-356335.16553451226, -356335.16553451226, -356335.16553451226]
mean = sum(values3) / len(values3)
manual_variance = sum((x - mean) ** 2 for x in values3) / len(values3)
print(f"Manually calculated mean: {mean}")
print(f"Manually calculated variance: {manual_variance}")
print()

# Test the current implementation's formula
print("Testing the implementation's formula directly:")
squares, total, n = chunk_result
x2 = float(squares)
x = float(total)
result = (x2 / n) - (x / n) ** 2
print(f"Sum of squares: {squares}")
print(f"Sum of values: {total}")
print(f"Count: {n}")
print(f"x2/n = {x2/n}")
print(f"(x/n)^2 = {(x/n)**2}")
print(f"Difference: {(x2/n) - (x/n)**2}")
print()

# Run property-based test with a few examples
print("Running property-based test with several examples:")

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=2, max_size=100))
@settings(max_examples=20)
def test_var_chunk_aggregate(values):
    chunk_result = var_chunk(values)
    aggregate_result = var_aggregate([chunk_result], ddof=0)

    mean = sum(values) / len(values)
    expected_variance = sum((x - mean) ** 2 for x in values) / len(values)

    try:
        assert aggregate_result >= 0, f"Variance cannot be negative: {aggregate_result}"
        assert math.isclose(aggregate_result, expected_variance,
                           rel_tol=1e-9, abs_tol=1e-9), \
               f"Variance mismatch: got {aggregate_result}, expected {expected_variance}"
    except AssertionError as e:
        print(f"FAILED: values={values[:3]}... error={e}")
        return

try:
    test_var_chunk_aggregate()
    print("Property test passed all examples")
except Exception as e:
    print(f"Property test failed: {e}")