from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")

try:
    assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
except AssertionError as e:
    print(f"AssertionError: {e}")

print("\nAdditional test cases:")
test_cases = [1125899906842624000, 1152921504606846975]
for test_n in test_cases:
    test_result = format_bytes(test_n)
    print(f"format_bytes({test_n}) → '{test_result}' ({len(test_result)} chars)")