import pandas as pd
from pandas.core.indexes.api import safe_sort_index

print("=" * 60)
print("Reproducing the bug from the report")
print("=" * 60)

# Create the exact index from the bug report
idx = pd.Index([0, 'a'], dtype='object')

# Call safe_sort_index
result = safe_sort_index(idx)

print("Input:", idx)
print("Input values:", idx.values)
print("Input dtype:", idx.dtype)
print()
print("Result:", result)
print("Result values:", result.values)
print("Result dtype:", result.dtype)
print()
print("Is result sorted?", result.is_monotonic_increasing or result.is_monotonic_decreasing)
print("Are input and result identical?", idx.equals(result))
print()
print("Docstring says:")
print(safe_sort_index.__doc__)
print()

# Let's also test what happens when we try to sort this directly
print("=" * 60)
print("What happens when we try to sort this index normally?")
print("=" * 60)
try:
    sorted_idx = pd.Index(sorted(idx))
    print("sorted() succeeded:", sorted_idx)
except TypeError as e:
    print("sorted() failed with TypeError:", e)

# Test with other mixed types
print()
print("=" * 60)
print("Testing with other mixed type scenarios")
print("=" * 60)

test_cases = [
    pd.Index([1, 2, 3]),  # All integers - should sort fine
    pd.Index(['a', 'b', 'c']),  # All strings - should sort fine
    pd.Index([1, 'a']),  # Mixed int/string
    pd.Index([1.0, 'a']),  # Mixed float/string
    pd.Index([True, 'a']),  # Mixed bool/string
    pd.Index([None, 1]),  # None with int
]

for test_idx in test_cases:
    result = safe_sort_index(test_idx)
    is_sorted = result.is_monotonic_increasing or result.is_monotonic_decreasing
    print(f"Input: {test_idx.values} -> Result: {result.values} | Sorted: {is_sorted}")