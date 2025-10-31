from dask.utils import format_bytes

# Test exactly where the boundary is for 10 vs 11 characters
test_values = []
base = 2**50  # 1 PiB

# Find exact boundary
for multiplier in range(998, 1002):
    n = multiplier * base
    result = format_bytes(n)
    test_values.append((n, result, len(result)))

print("Testing boundary values:")
for n, result, length in test_values:
    in_range = n < 2**60
    print(f"n={n:20} ({n/base:.2f} PiB), result='{result}', len={length}, n<2^60={in_range}")

# Check 2**60 boundary
print("\nTesting 2**60 boundary:")
for offset in [-2, -1, 0, 1]:
    try:
        n = 2**60 + offset
        result = format_bytes(n)
        print(f"n=2^60{offset:+2} = {n}, result='{result}', len={len(result)}")
    except:
        print(f"n=2^60{offset:+2} failed")

# Check the exact threshold calculation
k = 2**50  # Pi prefix
threshold = k * 0.9
print(f"\nPiB threshold: {threshold} ({threshold / 2**50:.2f} PiB)")
print(f"Max 3-digit PiB: {999.99 * 2**50}")
print(f"Min 4-digit PiB: {1000 * 2**50}")
print(f"2**60 = {2**60} = {2**60 / 2**50:.2f} PiB")