# Bug Report: pandas.arrays.IntegerArray Integer Overflow in fillna and setitem Operations

**Target**: `pandas.arrays.IntegerArray.fillna` and `pandas.arrays.IntegerArray.__setitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

IntegerArray's `fillna()` and `__setitem__()` methods crash with an obscure OverflowError when given integer values outside the int64 range, while the `insert()` method correctly raises a clear TypeError for the same input.

## Property-Based Test

```python
import numpy as np
import pandas.arrays as pa
from hypothesis import given, strategies as st, assume


@st.composite
def integer_array_with_na(draw):
    size = draw(st.integers(min_value=1, max_value=20))
    values = draw(st.lists(st.integers(min_value=-100, max_value=100), min_size=size, max_size=size))
    mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    assume(any(mask))
    return pa.IntegerArray(np.array(values, dtype='int64'), np.array(mask, dtype='bool'))


@given(integer_array_with_na(), st.integers())
def test_fillna_accepts_any_integer(arr, fill_value):
    result = arr.fillna(fill_value)
    assert not result.isna().any()


if __name__ == "__main__":
    test_fillna_accepts_any_integer()
```

<details>

<summary>
<strong>Failing input</strong>: <code>fill_value=9_223_372_036_854_775_808</code>
</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/38
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_fillna_accepts_any_integer FAILED                          [100%]

=================================== FAILURES ===================================
_______________________ test_fillna_accepts_any_integer ________________________

    @given(integer_array_with_na(), st.integers())
>   def test_fillna_accepts_any_integer(arr, fill_value):
                   ^^^

hypo.py:16:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:17: in test_fillna_accepts_any_integer
    result = arr.fillna(fill_value)
             ^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py:267: in fillna
    new_values[mask] = value
    ^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <IntegerArray>
[2, -3, -74, -68, -100, -10, -55, 86, <NA>]
Length: 9, dtype: Int64
key = array([False, False, False, False, False, False, False, False,  True])
value = 9223372036854775808

    def __setitem__(self, key, value) -> None:
        key = check_array_indexer(self, key)

        if is_scalar(value):
            if is_valid_na_for_dtype(value, self.dtype):
                self._mask[key] = True
            else:
                value = self._validate_setitem_value(value)
>               self._data[key] = value
                ^^^^^^^^^^^^^^^
E               OverflowError: Python int too large to convert to C long
E               Falsifying example: test_fillna_accepts_any_integer(
E                   arr=<IntegerArray>
E                   [2, -3, -74, -68, -100, -10, -55, 86, <NA>]
E                   Length: 9, dtype: Int64,
E                   fill_value=9_223_372_036_854_775_808,
E               )

/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py:316: OverflowError
=========================== short test summary info ============================
FAILED hypo.py::test_fillna_accepts_any_integer - OverflowError: Python int t...
============================== 1 failed in 0.39s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas.arrays as pa

# Create an IntegerArray with a masked value
arr = pa.IntegerArray(np.array([1, 2, 3]), np.array([False, True, False]))

# Value that exceeds int64 range (2^63)
overflow_value = 2**63

print("Testing fillna with overflow value:")
try:
    result = arr.fillna(overflow_value)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting __setitem__ with overflow value:")
try:
    arr_copy = arr.copy()
    arr_copy[1] = overflow_value
    print(f"Success: {arr_copy}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nFor comparison, testing insert() with overflow value:")
try:
    result = arr.insert(1, overflow_value)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
OverflowError crash in fillna() and __setitem__() vs clear TypeError in insert()
</summary>

```
Testing fillna with overflow value:
Error: OverflowError: Python int too large to convert to C long

Testing __setitem__ with overflow value:
Error: OverflowError: Python int too large to convert to C long

For comparison, testing insert() with overflow value:
Error: TypeError: cannot safely cast non-equivalent uint64 to int64
```
</details>

## Why This Is A Bug

This violates expected behavior for three key reasons:

1. **Inconsistent error handling within the same class**: The `insert()` method validates integer overflow and raises a clear TypeError ("cannot safely cast non-equivalent uint64 to int64"), while `fillna()` and `__setitem__()` crash with a cryptic OverflowError ("Python int too large to convert to C long").

2. **Poor user experience**: The OverflowError provides no context about the actual problem (value exceeding int64 range) or how to fix it. Users encountering this error have no indication it's related to the dtype's valid range.

3. **Incomplete validation**: The methods properly validate type constraints (rejecting floats/strings with clear TypeErrors) but fail to validate value range constraints for the int64 dtype, despite the insert() method demonstrating this validation is already implemented elsewhere in the codebase.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py:316` where `__setitem__` directly assigns the value without range validation. The `fillna()` method internally uses `__setitem__`, inheriting the same issue.

The `insert()` method uses `_from_sequence()` which properly validates the value range through internal casting checks, producing the informative TypeError.

Key observations:
- The failing value `9_223_372_036_854_775_808` is exactly 2^63, one beyond the maximum signed int64 value
- Both positive overflow (2^63) and negative underflow (-2^63 - 1) trigger the same error
- The int64 valid range is [-9_223_372_036_854_775_808, 9_223_372_036_854_775_807]

## Proposed Fix

Add overflow handling in the `__setitem__` method to provide a clear error message consistent with other validation in the class:

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -313,7 +313,17 @@ class BaseMaskedArray(ExtensionArray):
             if is_valid_na_for_dtype(value, self.dtype):
                 self._mask[key] = True
             else:
                 value = self._validate_setitem_value(value)
-                self._data[key] = value
+                try:
+                    self._data[key] = value
+                except OverflowError as err:
+                    # Provide a clear error message consistent with insert() method
+                    if value > np.iinfo(self._data.dtype).max:
+                        msg = f"cannot safely cast non-equivalent uint64 to {self.dtype.name.lower()}"
+                    else:
+                        msg = f"cannot safely cast value {value} to {self.dtype.name.lower()}"
+                    raise TypeError(msg) from err
                 self._mask[key] = False
             return
```