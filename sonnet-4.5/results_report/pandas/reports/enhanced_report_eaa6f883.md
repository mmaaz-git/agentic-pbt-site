# Bug Report: pandas.core.algorithms.unique Null Character String Collision

**Target**: `pandas.core.algorithms.unique`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `unique` function incorrectly treats strings containing only null characters (`\x00`) as identical to empty strings, failing to return all distinct values and causing silent data loss.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.algorithms import unique

@given(st.lists(st.text(min_size=0, max_size=10)))
@settings(max_examples=1000)
def test_unique_returns_all_distinct_values(values):
    values_array = np.array(values, dtype=object)
    uniques = unique(values_array)

    assert len(uniques) == len(set(values)), f"unique returned {len(uniques)} values but set has {len(set(values))}"
    assert set(uniques) == set(values), f"unique returned {set(uniques)} but expected {set(values)}"

if __name__ == "__main__":
    # Run the property-based test
    try:
        test_unique_returns_all_distinct_values()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Specifically test the failing case
    print("\n=== Testing specific failing case: ['', '\\x00'] ===")
    test_values = ['', '\x00']
    values_array = np.array(test_values, dtype=object)
    uniques = unique(values_array)

    print(f"Input: {[repr(v) for v in test_values]}")
    print(f"Expected unique count: {len(set(test_values))}")
    print(f"Actual unique count: {len(uniques)}")
    print(f"Uniques returned: {[repr(v) for v in uniques]}")

    try:
        assert len(uniques) == len(set(test_values))
        assert set(uniques) == set(test_values)
        print("This case passed!")
    except AssertionError:
        print("This case FAILED - confirming the bug!")
```

<details>

<summary>
**Failing input**: `values=['', '\x00']`
</summary>
```
Test failed: unique returned 1 values but set has 2

=== Testing specific failing case: ['', '\x00'] ===
Input: ["''", "'\\x00'"]
Expected unique count: 2
Actual unique count: 1
Uniques returned: ["''"]
This case FAILED - confirming the bug!
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.algorithms import unique, factorize

# Test case 1: Basic empty string vs null character
values = np.array(['', '\x00'], dtype=object)
uniques = unique(values)

print("=== Test Case 1: Basic empty string vs null character ===")
print(f"Input: {[repr(v) for v in values]}")
print(f"Expected unique count: 2")
print(f"Actual unique count: {len(uniques)}")
print(f"Uniques returned: {[repr(v) for v in uniques]}")
print(f"Python set sees {len(set(values))} unique values: {set([repr(v) for v in values])}")
print()

# Test case 2: Multiple null character strings
values2 = np.array(['\x00', '\x00\x00'], dtype=object)
uniques2 = unique(values2)

print("=== Test Case 2: Multiple null character strings ===")
print(f"Input: {[repr(v) for v in values2]}")
print(f"Expected unique count: 2")
print(f"Actual unique count: {len(uniques2)}")
print(f"Uniques returned: {[repr(v) for v in uniques2]}")
print(f"Python set sees {len(set(values2))} unique values: {set([repr(v) for v in values2])}")
print()

# Test case 3: Mixed with regular strings
values3 = np.array(['', '\x00', 'a'], dtype=object)
uniques3 = unique(values3)

print("=== Test Case 3: Mixed with regular strings ===")
print(f"Input: {[repr(v) for v in values3]}")
print(f"Expected unique count: 3")
print(f"Actual unique count: {len(uniques3)}")
print(f"Uniques returned: {[repr(v) for v in uniques3]}")
print(f"Python set sees {len(set(values3))} unique values: {set([repr(v) for v in values3])}")
print()

# Test case 4: Relationship to factorize
print("=== Test Case 4: Relationship to factorize ===")
values4 = np.array(['', '\x00'], dtype=object)
codes, fact_uniques = factorize(values4)

print(f"Input: {[repr(v) for v in values4]}")
print(f"Factorize codes: {codes}")
print(f"Factorize uniques: {[repr(v) for v in fact_uniques]}")
print(f"unique() result: {[repr(v) for v in unique(values4)]}")
print()

# Test Python's basic string operations
print("=== Python String Comparison ===")
print(f"'' == '\\x00': {'' == '\x00'}")
print(f"hash(''): {hash('')}")
print(f"hash('\\x00'): {hash('\x00')}")
test_dict = {'': 1}
print(f"'\\x00' in dict with '' as key: {'\x00' in test_dict}")
```

<details>

<summary>
Actual output showing data loss
</summary>
```
=== Test Case 1: Basic empty string vs null character ===
Input: ["''", "'\\x00'"]
Expected unique count: 2
Actual unique count: 1
Uniques returned: ["''"]
Python set sees 2 unique values: {"'\\x00'", "''"}

=== Test Case 2: Multiple null character strings ===
Input: ["'\\x00'", "'\\x00\\x00'"]
Expected unique count: 2
Actual unique count: 1
Uniques returned: ["'\\x00'"]
Python set sees 2 unique values: {"'\\x00\\x00'", "'\\x00'"}

=== Test Case 3: Mixed with regular strings ===
Input: ["''", "'\\x00'", "'a'"]
Expected unique count: 3
Actual unique count: 2
Uniques returned: ["''", "'a'"]
Python set sees 3 unique values: {"'\\x00'", "''", "'a'"}

=== Test Case 4: Relationship to factorize ===
Input: ["''", "'\\x00'"]
Factorize codes: [0 0]
Factorize uniques: ["''"]
unique() result: ["''"]

=== Python String Comparison ===
'' == '\x00': False
hash(''): 0
hash('\x00'): 3335806985461404630
'\x00' in dict with '' as key: False
```
</details>

## Why This Is A Bug

This violates the fundamental contract of the `unique()` function, which is documented to "Return unique values based on a hash table." The function fails to return all distinct values when null characters are present in strings.

Key evidence:
1. **Python correctly distinguishes these strings**: `'' != '\x00'` returns `False`, and they have different hash values (0 vs 3335806985461404630)
2. **Silent data loss**: When given `['', '\x00']`, the function returns only `['']`, losing the null character string entirely
3. **Affects multiple string patterns**: The bug occurs with `'\x00'`, `'\x00\x00'`, and any string containing only null bytes
4. **Shared bug with factorize**: Both `unique()` and `factorize()` exhibit identical incorrect behavior, indicating a systematic issue in the underlying hash table implementation

The documentation makes no mention of any limitations regarding null characters in strings, and Python natively supports null bytes in strings. This makes the bug particularly dangerous as users have no indication that their data is being silently discarded.

## Relevant Context

The bug appears to stem from the underlying C/Cython implementation in pandas' hash table code. The function flow is:
1. `unique()` calls `unique_with_mask()` at /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py:401
2. `unique_with_mask()` uses `_get_hashtable_algo()` to get the appropriate hash table at line 436
3. The hash table's `unique()` method is called at line 440

The issue likely occurs in the hash table implementation where null bytes (`\x00`) are being treated as C-style string terminators, causing strings containing only null bytes to be incorrectly hashed or compared.

Null characters can legitimately appear in:
- Binary data processing
- Database exports (especially from systems that use null-terminated strings)
- Network protocol data
- File format parsing
- Log files with control characters

## Proposed Fix

The fix requires modifying the underlying hash table implementation to properly handle Python strings containing null bytes. Since the exact C/Cython code location isn't visible in the Python source, here's the conceptual fix:

The hash table implementation needs to:
1. Use Python's string length information rather than relying on null-termination
2. Ensure string comparison uses the full byte sequence including null bytes
3. Hash the complete string content including embedded nulls

The issue is likely in pandas' C extension code (possibly in `pandas/_libs/hashtable.pyx` or similar), where string handling code may be using C string functions that treat `\x00` as a terminator instead of using Python's PyUnicode/PyBytes APIs that respect the stored length.