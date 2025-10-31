"""Test Python's built-in slice behavior for empty slices."""

test_cases = [
    (slice(1, 0, None), "slice(1, 0, None) - start > stop, positive step"),
    (slice(5, 2, None), "slice(5, 2, None) - start > stop, positive step"),
    (slice(10, -5, None), "slice(10, -5, None) - positive to negative"),
    (slice(0, 0, None), "slice(0, 0, None) - start == stop"),
    (slice(1, 1, None), "slice(1, 1, None) - start == stop"),
    (slice(-1, -5, None), "slice(-1, -5, None) - negative indices, backward"),
    (slice(5, 2, -1), "slice(5, 2, -1) - backward slice with negative step"),
    (slice(2, 5, -1), "slice(2, 5, -1) - forward indices with negative step"),
]

target = list(range(50))

print("Python's built-in slice behavior:")
print("-" * 60)

for slc, description in test_cases:
    result = target[slc]
    length = len(result)
    print(f"\n{description}")
    print(f"  Slice: {slc}")
    print(f"  Result: {result[:10] if len(result) > 10 else result}{'...' if len(result) > 10 else ''}")
    print(f"  Length: {length}")

# Check that Python NEVER returns negative length
print("\n" + "=" * 60)
print("Key observation: Python slices ALWAYS return non-negative lengths.")
print("An empty slice results in length 0, never negative.")