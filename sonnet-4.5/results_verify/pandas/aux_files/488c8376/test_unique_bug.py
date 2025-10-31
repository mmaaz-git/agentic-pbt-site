import pandas as pd
import numpy as np

# Test various cases to understand the unique() bug
test_cases = [
    ['0', '0\x00'],
    ['a', 'a\x00'],
    ['test', 'test\x00'],
    ['0', '0', '0\x00'],
    ['0\x00', '0'],
    ['a', 'b', 'a\x00'],
]

print("Testing pandas Index.unique() with null-terminated strings:")
print("=" * 60)

for test in test_cases:
    idx = pd.Index(test)
    unique_result = idx.unique()

    print(f"\nInput: {[repr(x) for x in test]}")
    print(f"Expected unique: {[repr(x) for x in sorted(set(test))]}")
    print(f"Actual unique:   {[repr(x) for x in unique_result]}")
    print(f"Data loss? {len(set(test)) != len(unique_result)}")

# Test with pure numpy
print("\n" + "=" * 60)
print("Testing numpy unique for comparison:")
test = ['0', '0', '0\x00']
np_array = np.array(test, dtype=object)
np_unique = np.unique(np_array)
print(f"Input: {[repr(x) for x in test]}")
print(f"numpy.unique: {[repr(x) for x in np_unique]}")

# Test pandas Series unique
print("\n" + "=" * 60)
print("Testing pandas Series.unique():")
series = pd.Series(['0', '0', '0\x00'])
series_unique = series.unique()
print(f"Input: {[repr(x) for x in series]}")
print(f"Series.unique: {[repr(x) for x in series_unique]}")