from dask.bag.chunk import var_chunk, var_aggregate

# Test case 1: Identical negative values (should have variance = 0)
values1 = [-356335.16553451226, -356335.16553451226, -356335.16553451226]
chunk_result1 = var_chunk(values1)
variance1 = var_aggregate([chunk_result1], ddof=0)

print("Test case 1: Three identical negative values")
print(f"Input: {values1}")
print(f"Computed variance: {variance1}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance1 < 0 else 'FAIL - Non-zero variance for identical values' if variance1 != 0 else 'PASS'}")
print()

# Test case 2: Identical positive values (should have variance = 0)
values2 = [259284.59765625, 259284.59765625, 259284.59765625]
chunk_result2 = var_chunk(values2)
variance2 = var_aggregate([chunk_result2], ddof=0)

print("Test case 2: Three identical positive values")
print(f"Input: {values2}")
print(f"Computed variance: {variance2}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance2 < 0 else 'FAIL - Non-zero variance for identical values' if variance2 != 0 else 'PASS'}")
print()

# Test case 3: Very large identical values
values3 = [1e15] * 10
chunk_result3 = var_chunk(values3)
variance3 = var_aggregate([chunk_result3], ddof=0)

print("Test case 3: Ten identical very large values")
print(f"Input: [1e15] * 10")
print(f"Computed variance: {variance3}")
print(f"Expected variance: 0.0")
print(f"Result: {'FAIL - Negative variance!' if variance3 < 0 else 'FAIL - Non-zero variance for identical values' if variance3 != 0 else 'PASS'}")
print()

# Test case 4: Simple values with known variance
values4 = [1, 2, 3, 4, 5]
chunk_result4 = var_chunk(values4)
variance4 = var_aggregate([chunk_result4], ddof=0)
expected4 = 2.0  # Variance of [1,2,3,4,5] is 2

print("Test case 4: Simple sequence [1,2,3,4,5]")
print(f"Input: {values4}")
print(f"Computed variance: {variance4}")
print(f"Expected variance: {expected4}")
print(f"Result: {'PASS' if abs(variance4 - expected4) < 1e-10 else 'FAIL - Incorrect variance'}")