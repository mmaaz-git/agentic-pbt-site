from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"Input: {n} (< 2**60: {n < 2**60})")
print(f"Output: '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Bug: {len(result)} > 10")