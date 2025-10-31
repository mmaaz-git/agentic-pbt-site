# Bug Report: pandas.api.types.union_categoricals Null Character Data Corruption

**Target**: `pandas.api.types.union_categoricals`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`union_categoricals` silently corrupts data by dropping the null character (`'\x00'`) from the merged categories, causing values that reference this missing category to become `NaN`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas as pd
from pandas.api import types

@given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
@example(['0', '\x000', '0', '\x00'])  # Known failing case
def test_union_categoricals_preserves_values(categories):
    """union_categoricals should preserve all unique values from input categoricals"""
    cat1 = pd.Categorical(categories[:len(categories)//2])
    cat2 = pd.Categorical(categories[len(categories)//2:])

    result = types.union_categoricals([cat1, cat2])

    expected_values = set(cat1.tolist() + cat2.tolist())
    result_values = set(result.tolist())

    assert expected_values == result_values, \
        f"union_categoricals did not preserve values: expected {expected_values}, got {result_values}"

if __name__ == "__main__":
    test_union_categoricals_preserves_values()
```

<details>

<summary>
**Failing input**: `categories=['0', '\x000', '0', '\x00']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/31
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_union_categoricals_preserves_values FAILED                 [100%]

=================================== FAILURES ===================================
___________________ test_union_categoricals_preserves_values ___________________

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
>   @example(['0', '\x000', '0', '\x00'])  # Known failing case
                   ^^^

hypo.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

categories = ['0', '\x000', '0', '\x00']

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
    @example(['0', '\x000', '0', '\x00'])  # Known failing case
    def test_union_categoricals_preserves_values(categories):
        """union_categoricals should preserve all unique values from input categoricals"""
        cat1 = pd.Categorical(categories[:len(categories)//2])
        cat2 = pd.Categorical(categories[len(categories)//2:])

        result = types.union_categoricals([cat1, cat2])

        expected_values = set(cat1.tolist() + cat2.tolist())
        result_values = set(result.tolist())

>       assert expected_values == result_values, \
            f"union_categoricals did not preserve values: expected {expected_values}, got {result_values}"
E       AssertionError: union_categoricals did not preserve values: expected {'0', '\x000', '\x00'}, got {'0', '\x000', nan}
E       assert {'\x00', '\x000', '0'} == {'0', '\x000', nan}
E
E         Extra items in the left set:
E         '\x00'
E         Extra items in the right set:
E         nan
E
E         Full diff:...
E
E         ...Full output truncated (6 lines hidden), use '-vv' to show
E       Falsifying explicit example: test_union_categoricals_preserves_values(
E           categories=['0', '\x000', '0', '\x00'],
E       )

hypo.py:17: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_union_categoricals_preserves_values - AssertionError: un...
============================== 1 failed in 0.92s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.types import union_categoricals

# Create categoricals with null character
cat1 = pd.Categorical(['0', '\x000'])  # String '0' followed by null character and '0'
cat2 = pd.Categorical(['0', '\x00'])   # String '0' and null character

print("Input categoricals:")
print(f"cat1.tolist(): {repr(cat1.tolist())}")
print(f"cat1.categories.tolist(): {repr(cat1.categories.tolist())}")
print(f"cat2.tolist(): {repr(cat2.tolist())}")
print(f"cat2.categories.tolist(): {repr(cat2.categories.tolist())}")

# Union the categoricals
result = union_categoricals([cat1, cat2])

print("\nResult after union:")
print(f"result.tolist(): {repr(result.tolist())}")
print(f"result.categories.tolist(): {repr(result.categories.tolist())}")

# Check preservation of null character
print("\nValidation:")
print(f"'\\x00' in cat2.tolist(): {'\x00' in cat2.tolist()}")
print(f"'\\x00' in result.tolist(): {'\x00' in result.tolist()}")

# This should pass but will fail if the bug exists
assert '\x00' in cat2.tolist(), "\\x00 not found in cat2"
assert '\x00' in result.tolist(), "\\x00 not found in result, but it should be preserved!"
```

<details>

<summary>
AssertionError: \x00 not found in result, but it should be preserved!
</summary>
```
Input categoricals:
cat1.tolist(): ['0', '\x000']
cat1.categories.tolist(): ['\x000', '0']
cat2.tolist(): ['0', '\x00']
cat2.categories.tolist(): ['\x00', '0']

Result after union:
result.tolist(): ['0', '\x000', '0', nan]
result.categories.tolist(): ['\x000', '0']

Validation:
'\x00' in cat2.tolist(): True
'\x00' in result.tolist(): False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 28, in <module>
    assert '\x00' in result.tolist(), "\\x00 not found in result, but it should be preserved!"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: \x00 not found in result, but it should be preserved!
```
</details>

## Why This Is A Bug

The `union_categoricals` function is documented to combine categorical arrays while preserving all unique categories. However, it silently drops the null character (`'\x00'`) during the category merging process. This violates the contract because:

1. **Data Loss**: The null character is a valid Unicode character that should be preserved in string data. When `cat2` contains `'\x00'` as a category, it should appear in the result's categories.

2. **Silent Corruption**: Values that reference the missing `'\x00'` category become `NaN` without warning. In the example, the second element from `cat2` becomes `nan` instead of `'\x00'`.

3. **Inconsistent with NumPy**: NumPy's `unique` function correctly identifies `'\x00'` and `'\x000'` as distinct values, but pandas' `Index.unique()` incorrectly merges them.

## Relevant Context

The bug originates in pandas' `Index.unique()` method, which incorrectly treats `'\x00'` and `'\x000'` as the same value:

```python
import pandas as pd
import numpy as np

# NumPy correctly identifies 3 unique values
arr = np.array(['\x000', '0', '\x00', '0'], dtype=object)
print(np.unique(arr))  # Output: ['\x00' '\x000' '0']

# Pandas incorrectly identifies only 2 unique values
idx = pd.Index(['\x000', '0', '\x00', '0'])
print(idx.unique().tolist())  # Output: ['\x000', '0']
```

This affects `union_categoricals` at line 328-329 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/concat.py`:

```python
cats = first.categories.append([c.categories for c in to_union[1:]])
categories = cats.unique()  # Bug: unique() drops '\x00'
```

## Proposed Fix

The root cause appears to be in pandas' unique algorithm for object arrays, likely in the hashtable implementation that doesn't properly handle null characters. A comprehensive fix would require:

1. Fixing the `pandas.core.algorithms.unique` function to properly handle null characters in object arrays
2. Ensuring the underlying hashtable implementation distinguishes between strings containing null characters
3. Adding test cases for null characters and other special Unicode characters

As a workaround, users should avoid using null characters in categorical data until this bug is fixed.