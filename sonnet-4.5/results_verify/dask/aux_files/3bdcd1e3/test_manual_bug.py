from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")

assert n < 2**60
print(f"Assertion n < 2**60: passed")
assert len(result) > 10
print(f"Assertion len(result) > 10: passed (length = {len(result)})")

# Test with more edge cases
test_cases = [
    1000 * 2**50,  # Exactly 1000 PiB
    999 * 2**50,   # Just under 1000 PiB
    1001 * 2**50,  # Just over 1000 PiB
    2**60 - 1,     # Maximum value mentioned in docstring
]

print("\nAdditional test cases:")
for test_n in test_cases:
    test_result = format_bytes(test_n)
    print(f"format_bytes({test_n:20}) = '{test_result:12}' (length: {len(test_result)})")