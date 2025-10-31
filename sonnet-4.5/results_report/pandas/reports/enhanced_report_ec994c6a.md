# Bug Report: pandas.core.arrays.Categorical Null Character Data Corruption

**Target**: `pandas.core.arrays.Categorical`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Categorical silently corrupts data by treating the null character `'\x00'` as equivalent to the empty string `''`, causing distinct string values to be incorrectly merged into a single category and resulting in data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arrays

@given(st.lists(st.one_of(st.text(min_size=0, max_size=5), st.none()), min_size=1, max_size=30))
@settings(max_examples=500)
def test_categorical_codes_bounds(values):
    cat = arrays.Categorical(values)

    assert len(cat) == len(values)
    assert len(cat.codes) == len(values)

    for i, code in enumerate(cat.codes):
        if values[i] is None:
            assert code == -1, f"NA should have code -1 at index {i}, got {code}"
        else:
            assert 0 <= code < len(cat.categories), f"Code {code} out of bounds at index {i}"
            assert cat.categories[code] == values[i], f"Code mapping wrong at {i}"

if __name__ == "__main__":
    test_categorical_codes_bounds()
```

<details>

<summary>
**Failing input**: `values=['', '\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 20, in <module>
    test_categorical_codes_bounds()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 5, in test_categorical_codes_bounds
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 17, in test_categorical_codes_bounds
    assert cat.categories[code] == values[i], f"Code mapping wrong at {i}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Code mapping wrong at 1
Falsifying example: test_categorical_codes_bounds(
    values=['', '\x00'],
)
```
</details>

## Reproducing the Bug

```python
import pandas.core.arrays as arrays

values = ['', '\x00']
cat = arrays.Categorical(values)

print(f"Input: {repr(values)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes: {list(cat.codes)}")

reconstructed = [cat.categories[c] if c != -1 else None for c in cat.codes]
print(f"Original:      {repr(values)}")
print(f"Reconstructed: {repr(reconstructed)}")
print(f"Data preserved: {values == reconstructed}")
```

<details>

<summary>
Output showing data corruption
</summary>
```
Input: ['', '\x00']
Categories: ['']
Codes: [np.int8(0), np.int8(0)]
Original:      ['', '\x00']
Reconstructed: ['', '']
Data preserved: False
```
</details>

## Why This Is A Bug

This violates the fundamental contract of pandas Categorical that distinct input values should map to distinct categories. In Python, `''` (empty string, length 0) and `'\x00'` (null character, length 1) are completely different values (`'' == '\x00'` returns `False`). However, pandas incorrectly merges them into a single category.

The bug causes silent data corruption where:
1. Two distinct string values are incorrectly treated as identical
2. The null character `'\x00'` is lost and replaced with `''`
3. Data cannot be round-tripped through Categorical without loss
4. Both values incorrectly map to the same category code (0)

This violates the documented behavior that Categorical preserves distinct values and can represent values "of any dtype". The documentation makes no exception for null characters and implies all distinct values should be preserved as separate categories.

## Relevant Context

This bug appears to originate in pandas' string deduplication logic, specifically in the `factorize` function which is used internally by Categorical to identify unique values. Testing shows:

```python
import pandas.core.algorithms as algorithms
codes, uniques = algorithms.factorize(['', '\x00'])
# Returns: codes=[0, 0], uniques=['']
```

Related pandas issues confirm this is a known problem area:
- Issue #34551 (open): factorize() and drop_duplicates() incorrectly handle strings with null bytes
- Issue #61189 (closed as duplicate): null bytes not preserved in CategoricalIndex/MultiIndex
- Issues #14012, #19886: CSV reading issues with null characters

Real-world scenarios where this bug causes problems:
- Processing binary data with embedded null bytes
- C string interoperability where `\x00` has special meaning
- Network protocols using null bytes as delimiters
- Data serialization formats that distinguish between empty and null-terminated strings

## Proposed Fix

The bug likely stems from pandas using C-style string comparisons that treat `'\x00'` as a string terminator. A high-level fix would involve ensuring pandas' string handling uses Python's native string equality which correctly distinguishes `'\x00'` from `''`. The fix would need to be applied to the underlying string factorization and hashing logic, potentially in the Cython/C extensions that handle string operations for performance.

Key areas that likely need modification:
1. The `factorize` function in pandas.core.algorithms
2. String hashing functions used for deduplication
3. Any C/Cython code that uses strcmp or similar C string functions instead of Python string comparison

Without access to modify the pandas source directly, users can work around this by:
- Escaping null bytes before creating Categoricals
- Using a different data structure when null bytes are present
- Pre-processing data to replace `\x00` with a placeholder value