from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print(f"Is n < 2**60? {n < 2**60}")
print(f"2**60 = {2**60}")

# This should pass according to the docstring guarantee
assert len(result) <= 10, f"Length {len(result)} exceeds guaranteed maximum of 10 characters"