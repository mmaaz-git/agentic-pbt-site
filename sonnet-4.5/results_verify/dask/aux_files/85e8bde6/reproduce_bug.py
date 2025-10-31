from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"format_bytes(n) = '{result}'")
print(f"len(result) = {len(result)}")

# Additional test values around the boundary
test_values = [
    1_125_894_277_343_089_729,  # The reported failing case
    1_125_899_906_842_624_000,  # 1000 * 2**50 exactly
    1_125_899_906_842_623_999,  # Just below 1000 * 2**50
    999 * 2**50,                # Just below the threshold
    1000 * 2**50,               # At the threshold
    1001 * 2**50,               # Just above
]

print("\n--- Testing various values near the boundary ---")
for val in test_values:
    result = format_bytes(val)
    print(f"n = {val:25} | n < 2**60: {val < 2**60} | result: '{result:12}' | len: {len(result)}")