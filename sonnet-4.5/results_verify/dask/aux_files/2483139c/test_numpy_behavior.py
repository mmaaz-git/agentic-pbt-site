import numpy as np

# Test various edge cases with numpy to understand expected behavior
test_cases = [
    # (array_size, start, stop, step)
    (1, -2, -2, -1),
    (3, -5, -5, -2),
    (5, 3, 3, -1),
    (5, -1, -1, -1),
    (10, 5, 5, -2),
]

print("Testing NumPy slicing behavior for empty slices:")
print("=" * 60)

for size, start, stop, step in test_cases:
    arr = np.arange(size)
    idx = slice(start, stop, step)
    result = arr[idx]

    # Also test what indices() returns
    indices = idx.indices(size)

    print(f"Array size {size}, slice({start}, {stop}, {step}):")
    print(f"  Result: {result} (length: {len(result)})")
    print(f"  indices({size}): {indices}")
    print(f"  Is empty: {len(result) == 0}")
    print()

# Test Python's list slicing behavior
print("\nTesting Python list slicing behavior:")
print("=" * 60)

for size, start, stop, step in test_cases:
    lst = list(range(size))
    idx = slice(start, stop, step)
    result = lst[idx]

    print(f"List size {size}, slice({start}, {stop}, {step}):")
    print(f"  Result: {result} (length: {len(result)})")
    print(f"  Is empty: {len(result) == 0}")
    print()