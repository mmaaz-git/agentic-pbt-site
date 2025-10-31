import dask.utils

# Let's understand the implementation better
# The function uses >= k * 0.9 for thresholds

print("Understanding the implementation:")
print("The function checks if n >= k * 0.9 for each unit")
print()

# PiB threshold: 2**50 * 0.9
pib_threshold = 2**50 * 0.9
print(f"PiB threshold: {pib_threshold}")
print(f"1000 PiB in bytes: {1000 * 2**50}")
print(f"1024 PiB in bytes: {1024 * 2**50}")
print()

# Test edge cases around 1000 PiB
test_values = [
    999.99 * 2**50,
    1000.00 * 2**50,
    1000.01 * 2**50,
    1023.99 * 2**50,
    1024.00 * 2**50,
]

for val in test_values:
    result = dask.utils.format_bytes(int(val))
    print(f"{val/2**50:.2f} PiB -> format_bytes({int(val)}) = '{result}' (len={len(result)})")