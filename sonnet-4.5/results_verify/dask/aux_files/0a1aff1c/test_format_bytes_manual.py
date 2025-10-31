import dask.utils

# Test the specific failing example from bug report
n = 1125899906842624000
result = dask.utils.format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print()

# Test additional examples from bug report
test_cases = [
    (1124774006935781376, "999 PiB"),
    (1125899906842624000, "1000 PiB"),
    (1151795604700004352, "1023 PiB"),
]

print("Additional examples:")
for n, expected_prefix in test_cases:
    result = dask.utils.format_bytes(n)
    print(f"format_bytes({n:20d}) = {result:12s} (length={len(result):2d}) {'✓' if len(result) <= 10 else '✗'}")

# Check 2**60
print(f"\n2**60 = {2**60}")
print(f"Test value {1125899906842624000} < 2**60: {1125899906842624000 < 2**60}")