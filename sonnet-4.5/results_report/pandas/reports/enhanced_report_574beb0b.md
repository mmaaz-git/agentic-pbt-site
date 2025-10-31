# Bug Report: pandas.Categorical Null Character Data Corruption

**Target**: `pandas.Categorical`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.Categorical` silently corrupts data by treating strings that differ only by trailing null characters (`\x00`) as identical, causing data loss without warning.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings, example


@given(arr=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=50))
@settings(max_examples=500)
@example(arr=['', '\x00'])  # Add explicit failing case
def test_categorical_round_trip(arr):
    cat = pd.Categorical(arr)
    reconstructed = cat.categories[cat.codes]

    assert list(reconstructed) == arr, \
        f"Categorical round-trip failed: original {arr}, reconstructed {list(reconstructed)}"

# Run the test
if __name__ == "__main__":
    import traceback
    try:
        test_categorical_round_trip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error:")
        print(str(e))
        traceback.print_exc()
    except Exception as e:
        print(f"Test failed with error:")
        print(str(e))
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `['', '\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/./hypo.py", line 22, in <module>
    test_categorical_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/./hypo.py", line 9, in test_categorical_round_trip
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/./hypo.py", line 15, in test_categorical_round_trip
    assert list(reconstructed) == arr, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Categorical round-trip failed: original ['', '\x00'], reconstructed ['', '']
Falsifying explicit example: test_categorical_round_trip(
    arr=['', '\x00'],
)
Test failed with assertion error:
Categorical round-trip failed: original ['', '\x00'], reconstructed ['', '']
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd

# Test case 1: Empty string vs null character
print("Test 1: Empty string vs null character")
arr = ['', '\x00']
cat = pd.Categorical(arr)

print(f"Input:      {repr(arr)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes:      {cat.codes.tolist()}")

reconstructed = list(cat.categories[cat.codes])
print(f"Output:     {repr(reconstructed)}")
print(f"Match:      {reconstructed == arr}")
print()

# Test case 2: String with and without trailing null
print("Test 2: String with and without trailing null")
arr2 = ['a', 'a\x00']
cat2 = pd.Categorical(arr2)

print(f"Input:      {repr(arr2)}")
print(f"Categories: {repr(list(cat2.categories))}")
print(f"Codes:      {cat2.codes.tolist()}")

reconstructed2 = list(cat2.categories[cat2.codes])
print(f"Output:     {repr(reconstructed2)}")
print(f"Match:      {reconstructed2 == arr2}")
print()

# Test case 3: Null character vs other control character (this works correctly)
print("Test 3: Null character vs other control character")
arr3 = ['\x00', '\x01']
cat3 = pd.Categorical(arr3)

print(f"Input:      {repr(arr3)}")
print(f"Categories: {repr(list(cat3.categories))}")
print(f"Codes:      {cat3.codes.tolist()}")

reconstructed3 = list(cat3.categories[cat3.codes])
print(f"Output:     {repr(reconstructed3)}")
print(f"Match:      {reconstructed3 == arr3}")
```

<details>

<summary>
Silent data corruption demonstrated across multiple test cases
</summary>
```
Test 1: Empty string vs null character
Input:      ['', '\x00']
Categories: ['']
Codes:      [0, 0]
Output:     ['', '']
Match:      False

Test 2: String with and without trailing null
Input:      ['a', 'a\x00']
Categories: ['a']
Codes:      [0, 0]
Output:     ['a', 'a']
Match:      False

Test 3: Null character vs other control character
Input:      ['\x00', '\x01']
Categories: ['\x00', '\x01']
Codes:      [0, 1]
Output:     ['\x00', '\x01']
Match:      True
```
</details>

## Why This Is A Bug

This is a clear data integrity bug that violates fundamental expectations:

1. **Silent Data Corruption**: The null character `\x00` is a valid Python string character that should be preserved. Pandas is silently changing user data without any warning, violating the principle that distinct input values should remain distinct.

2. **Violates Python String Semantics**: In Python, `'' != '\x00'` and `'a' != 'a\x00'`. These are distinct string objects with different lengths and content. Pandas should respect Python's string semantics.

3. **Breaks Round-Trip Property**: Creating a Categorical from an array should preserve the array's values. The operation `list(pd.Categorical(arr).categories[pd.Categorical(arr).codes])` should equal `arr` for all valid input arrays.

4. **Inconsistent Behavior**: The bug only affects strings where one is a prefix of another with trailing null bytes. It works correctly for `['\x00', '\x01']` but fails for `['', '\x00']` and `['a', 'a\x00']`.

5. **Root Cause Identified**: The issue stems from pandas' `StringHashTable` implementation which incorrectly treats strings differing only by trailing nulls as identical, likely due to C-style string comparison that stops at null bytes.

## Relevant Context

The bug occurs at the factorization level in pandas' internal hash table implementation. Testing reveals:

- `pandas._libs.hashtable.StringHashTable` incorrectly deduplicates strings with trailing nulls
- `pandas._libs.hashtable.PyObjectHashTable` handles the same data correctly
- The issue affects all pandas operations that rely on factorization, not just Categorical

This could impact:
- Data analysis involving binary protocols or control characters
- Text processing where null bytes have semantic meaning
- Any workflow where data integrity must be preserved exactly

The bug is particularly dangerous because it's silent - users have no indication their data is being corrupted.

## Proposed Fix

The fix requires modifying the StringHashTable implementation to properly handle null bytes in strings. As a workaround, pandas could fall back to PyObjectHashTable when null bytes are detected, though this would have performance implications.

Without access to the C/Cython source, the exact fix would involve ensuring the string comparison in StringHashTable uses the full Python string length rather than stopping at null bytes (avoiding C-style string semantics).

A temporary workaround for users:
- Encode strings with null bytes using a different representation before creating Categoricals
- Use `.astype('category')` on a Series with object dtype to force different internal handling
- Be aware of this limitation when processing binary data or special characters