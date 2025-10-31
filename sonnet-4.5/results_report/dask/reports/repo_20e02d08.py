from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"format_bytes(n) = '{result}'")
print(f"len(result) = {len(result)}")

# Additional boundary tests
print("\n--- Additional tests around the boundary ---")
boundary = 1000 * 2**50
for offset in [-1, 0, 1]:
    test_n = boundary + offset
    test_result = format_bytes(test_n)
    print(f"n = {test_n:,}")
    print(f"format_bytes(n) = '{test_result}'")
    print(f"len(result) = {len(test_result)}")
    print()