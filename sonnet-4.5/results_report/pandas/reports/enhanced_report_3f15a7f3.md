# Bug Report: pandas.core.algorithms String Collision in factorize/unique/duplicated

**Target**: `pandas.core.algorithms.factorize`, `pandas.core.algorithms.unique`, `pandas.core.algorithms.duplicated`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The StringHashTable implementation in pandas incorrectly treats the empty string `''` and strings containing null character followed by other characters like `'\x000'` as identical values, causing data corruption in `factorize()`, `unique()`, and `duplicated()` functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core import algorithms as alg

@given(st.lists(st.text(min_size=0, max_size=10)))
def test_factorize_round_trip(values):
    if len(values) == 0:
        return

    arr = np.array(values)
    codes, uniques = alg.factorize(arr)

    reconstructed = uniques.take(codes[codes >= 0])
    original_without_na = arr[codes >= 0]

    if len(reconstructed) > 0 and len(original_without_na) > 0:
        assert all(a == b for a, b in zip(reconstructed, original_without_na))
```

<details>

<summary>
**Failing input**: `['', '\x000']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 22, in <module>
    test_factorize_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 6, in test_factorize_round_trip
    def test_factorize_round_trip(values):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 17, in test_factorize_round_trip
    assert all(a == b for a, b in zip(reconstructed, original_without_na))
           ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_factorize_round_trip(
    values=['', '\x000'],
)
Test failed!

Testing specific failing case: ['', '\x000']
Specific test failed: reconstructed values don't match originals
  Original: [np.str_(''), np.str_('\x000')]
  Reconstructed: [np.str_(''), np.str_('')]
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core import algorithms as alg

values = ['', '\x000']
arr = np.array(values)

codes, uniques = alg.factorize(arr)
print(f"codes: {codes}")
print(f"uniques: {uniques}")

print(f"\nExpected: codes=[0, 1], uniques=['', '\\x000']")
print(f"Actual: codes={list(codes)}, uniques={list(uniques)}")
print(f"\nBoth strings map to same code despite being different:")
print(f"  '' == '\\x000': {'' == '\x000'}")
print(f"  len('')={len('')}, len('\\x000')={len('\x000')}")

unique_vals = alg.unique(arr)
print(f"\nunique() also fails:")
print(f"  Expected 2 unique values, got {len(unique_vals)}")

dup_mask = alg.duplicated(arr, keep='first')
print(f"\nduplicated() also fails:")
print(f"  Mask: {dup_mask}")
print(f"  Both marked as non-duplicates despite unique() returning 1 value")
```

<details>

<summary>
Output showing data corruption
</summary>
```
codes: [0 0]
uniques: ['']

Expected: codes=[0, 1], uniques=['', '\x000']
Actual: codes=[np.int64(0), np.int64(0)], uniques=[np.str_('')]

Both strings map to same code despite being different:
  '' == '\x000': False
  len('')=0, len('\x000')=2

unique() also fails:
  Expected 2 unique values, got 1

duplicated() also fails:
  Mask: [False False]
  Both marked as non-duplicates despite unique() returning 1 value
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of these pandas functions:

1. **factorize() contract violation**: The documentation explicitly states that "uniques.take(codes) will have the same values as values" (line 661 in algorithms.py). This property is violated because:
   - Input: `values[1] = '\x000'` (a 2-character string)
   - Output: `uniques.take(codes[1]) = uniques[0] = ''` (empty string)
   - These are not equal: `'\x000' != ''`

2. **Data corruption**: Two distinct strings are collapsed into one, causing silent data loss. Python and NumPy both correctly distinguish these strings:
   - Python: `'' == '\x000'` returns `False`
   - Python: `hash('')` and `hash('\x000')` produce different values
   - NumPy: `np.unique(['', '\x000'])` correctly returns both values

3. **Internal inconsistency**: The `duplicated()` function marks both values as non-duplicates (returning `[False, False]`), yet `unique()` returns only one value. This is logically inconsistent.

4. **Unexpected behavior**: The bug only occurs when StringHashTable is used. When PyObjectHashTable is used instead, the functions work correctly. The selection of StringHashTable happens automatically based on dtype detection, making this bug unpredictable for users.

## Relevant Context

The root cause is in the StringHashTable implementation (pandas._libs.hashtable.StringHashTable). When pandas detects that an array contains only strings (via `lib.is_string_array()`), it uses StringHashTable for performance reasons instead of the general PyObjectHashTable.

Testing confirms:
- **StringHashTable** (buggy): `factorize(['', '\x000'])` → codes=[0, 0], uniques=['']
- **PyObjectHashTable** (correct): `factorize(['', '\x000'])` → codes=[0, 1], uniques=['', '\x000']

The bug affects these code paths:
- `pandas/core/algorithms.py:632` - Main factorize function
- `pandas/core/algorithms.py:795` - Calls factorize_array for numpy arrays
- `pandas/core/algorithms.py:594` - Uses hashtable selected by _get_hashtable_algo
- `pandas/core/algorithms.py:297` - StringHashTable selected when lib.is_string_array returns True

Documentation links:
- [pandas.factorize documentation](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html)
- [User guide on factorize](https://pandas.pydata.org/docs/user_guide/reshaping.html#factorize)

## Proposed Fix

The bug is in the C/Cython implementation of StringHashTable. Two potential fixes:

1. **Quick workaround** (in algorithms.py): Don't use StringHashTable for arrays that might contain null characters:

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -293,7 +293,10 @@ def _check_object_for_strings(values: np.ndarray) -> str:
     if ndtype == "object":
         # it's cheaper to use a String Hash Table than Object; we infer
         # including nulls because that is the only difference between
         # StringHashTable and ObjectHashtable
-        if lib.is_string_array(values, skipna=False):
+        # Note: StringHashTable has a bug with null characters, so check for them
+        if lib.is_string_array(values, skipna=False) and not any(
+            isinstance(v, str) and '\x00' in v for v in values
+        ):
             ndtype = "string"
     return ndtype
```

2. **Proper fix** (in pandas/_libs/hashtable.pyx): Fix the StringHashTable implementation to properly handle strings with null characters. This likely involves:
   - Ensuring the hash function uses the full string length, not stopping at null characters
   - Ensuring string comparison uses proper Python string equality, not C-style strcmp

The proper fix requires investigation of the StringHashTable Cython implementation to identify where null character handling fails.