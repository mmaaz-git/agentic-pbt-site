# Bug Report: ArrowExtensionArray.insert() Fails When Inserting Non-None Values Into Null-Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.insert`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `insert()` method raises `ArrowNotImplementedError` when attempting to insert a non-None value into an ArrowExtensionArray that contains only None values (null type), instead of successfully inserting the value or providing a meaningful error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_int_array = st.lists(
    st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
    min_size=0,
    max_size=100
).map(lambda x: ArrowExtensionArray(pa.array(x)))

@given(arrow_int_array, st.integers(min_value=0, max_value=50), st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_insert_delete_inverse(arr, loc_int, item):
    assume(len(arr) > 0)
    loc = loc_int % (len(arr) + 1)

    inserted = arr.insert(loc, item)
    deleted = inserted.delete(loc)

    assert arr.equals(deleted), "delete(insert(arr, loc, item), loc) should equal arr"

if __name__ == "__main__":
    test_insert_delete_inverse()
```

<details>

<summary>
**Failing input**: `arr=(lambda x: ArrowExtensionArray(pa.array(x)))([None]), loc_int=0, item=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 23, in <module>
    test_insert_delete_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 12, in test_insert_delete_inverse
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 17, in test_insert_delete_inverse
    inserted = arr.insert(loc, item)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/base.py", line 2116, in insert
    item_arr = type(self)._from_sequence([item], dtype=self.dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 312, in _from_sequence
    pa_array = cls._box_pa_array(scalars, pa_type=pa_type, copy=copy)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 528, in _box_pa_array
    pa_array = pa_array.cast(pa_type)
  File "pyarrow/array.pxi", line 1102, in pyarrow.lib.Array.cast
  File "/home/npc/.local/lib/python3.13/site-packages/pyarrow/compute.py", line 410, in cast
    return call_function("cast", [arr], options, memory_pool)
  File "pyarrow/_compute.pyx", line 612, in pyarrow._compute.call_function
  File "pyarrow/_compute.pyx", line 407, in pyarrow._compute.Function.call
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowNotImplementedError: Unsupported cast from int64 to null using function cast_null
Falsifying example: test_insert_delete_inverse(
    # The test always failed when commented parts were varied together.
    arr=(lambda x: ArrowExtensionArray(pa.array(x)))([None]),
    loc_int=0,  # or any other generated value
    item=0,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:503
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:524
```
</details>

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray containing only None
arr = ArrowExtensionArray(pa.array([None]))

# Try to insert a non-None value
try:
    result = arr.insert(0, 42)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Error: ArrowNotImplementedError when inserting into null-type array
</summary>
```
Error: ArrowNotImplementedError: Unsupported cast from int64 to null using function cast_null
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Documentation Violation**: The `ExtensionArray.insert()` documentation (in `pandas/core/arrays/base.py:2095`) states that the method should "Insert an item at the given position" and if the item cannot be held in the array, it should raise either `ValueError` or `TypeError`. Instead, it raises `ArrowNotImplementedError`, which is neither of the documented exception types.

2. **Inconsistent Behavior**: When creating an array with mixed None and non-None values like `ArrowExtensionArray(pa.array([None, 42]))`, PyArrow correctly infers the type as `int64`. However, when starting with only None values, it creates a `null` type array that cannot accept any non-None values later via insert().

3. **Breaks Insert Contract**: The insert method is a fundamental array operation that should either succeed or provide a clear, user-friendly error. The current error message about "Unsupported cast from int64 to null using function cast_null" is a low-level PyArrow implementation detail that doesn't help users understand the issue.

4. **Common Use Case**: Starting with an empty or all-None array and progressively adding values is a realistic and common pattern in data processing workflows.

## Relevant Context

The error occurs in the `insert()` method at line 2116 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/base.py`:
```python
item_arr = type(self)._from_sequence([item], dtype=self.dtype)
```

When `self.dtype` is `null` (because the array contains only None values), and `item` is a non-None value, `_from_sequence` creates an array with the inferred type (e.g., `int64`) and then attempts to cast it to `null` type. This cast operation fails in PyArrow because you cannot cast from a concrete type to the null type.

The call stack shows the failure happens in `_box_pa_array` at line 528 of `pandas/core/arrays/arrow/array.py` when attempting:
```python
pa_array = pa_array.cast(pa_type)
```

PyArrow documentation: https://arrow.apache.org/docs/python/generated/pyarrow.null.html
ExtensionArray documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionArray.html

## Proposed Fix

The issue can be fixed by detecting when inserting a non-None value into a null-type array and handling this case specially. Here's a patch for `pandas/core/arrays/base.py`:

```diff
def insert(self, loc: int, item) -> Self:
    """
    Insert an item at the given position.
    ...
    """
    loc = validate_insert_loc(loc, len(self))
+
+   # Handle insertion into null-type arrays (ArrowExtensionArray specific)
+   if hasattr(self, '_pa_array'):
+       import pyarrow as pa
+       if pa.types.is_null(self._pa_array.type) and item is not None:
+           # Create a new array with the appropriate type inferred from the item
+           item_arr = type(self)._from_sequence([item])
+           # Convert the null array to the same type as the item
+           converted_arr = pa.array([None] * len(self), type=item_arr._pa_array.type)
+           temp_arr = type(self)(converted_arr)
+           # Now perform the insert with compatible types
+           return type(self)._concat_same_type([temp_arr[:loc], item_arr, temp_arr[loc:]])
+
    item_arr = type(self)._from_sequence([item], dtype=self.dtype)
    return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])
```