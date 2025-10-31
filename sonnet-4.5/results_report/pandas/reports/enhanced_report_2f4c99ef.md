# Bug Report: pandas.core.arrays.arrow.ArrowExtensionArray.fillna raises uncaught ArrowInvalid on null-type arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `fillna()` method on ArrowExtensionArray crashes with an uncaught `pyarrow.lib.ArrowInvalid` exception when called on arrays containing only None values (which have PyArrow's null type), instead of catching this exception and converting it to a user-friendly TypeError.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(st.data())
@settings(max_examples=500)
def test_arrow_extension_array_fillna_length(data):
    values = data.draw(st.lists(st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()), min_size=1, max_size=100))
    arr = ArrowExtensionArray(pa.array(values))
    fill_value = data.draw(st.integers(min_value=-1000, max_value=1000))

    result = arr.fillna(fill_value)

    assert len(result) == len(arr)

if __name__ == "__main__":
    # Run the test
    test_arrow_extension_array_fillna_length()
```

<details>

<summary>
**Failing input**: `values=[None], fill_value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 21, in <module>
    test_arrow_extension_array_fillna_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 9, in test_arrow_extension_array_fillna_length
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 15, in test_arrow_extension_array_fillna_length
    result = arr.fillna(fill_value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1160, in fillna
    fill_value = self._box_pa(value, pa_type=self._pa_array.type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 407, in _box_pa
    return cls._box_pa_scalar(value, pa_type)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 443, in _box_pa_scalar
    pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
  File "pyarrow/scalar.pxi", line 1599, in pyarrow.lib.scalar
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Invalid null value
Falsifying example: test_arrow_extension_array_fillna_length(
    data=data(...),
)
Draw 1: [None]
Draw 2: 0
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:1161
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray with only None values
arr = ArrowExtensionArray(pa.array([None]))

# Try to fill NA values with 0
# This should raise an ArrowInvalid error instead of being caught properly
result = arr.fillna(0)
print(result)
```

<details>

<summary>
ArrowInvalid: Invalid null value
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/repo.py", line 12, in <module>
    result = arr.fillna(0)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1160, in fillna
    fill_value = self._box_pa(value, pa_type=self._pa_array.type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 407, in _box_pa
    return cls._box_pa_scalar(value, pa_type)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 443, in _box_pa_scalar
    pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
  File "pyarrow/scalar.pxi", line 1599, in pyarrow.lib.scalar
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Invalid null value
```
</details>

## Why This Is A Bug

This violates expected behavior for the following reasons:

1. **Incomplete exception handling**: The fillna method at line 1161 in array.py has a try-except block that catches `pa.ArrowTypeError` to handle type conversion failures gracefully. However, PyArrow raises `pa.ArrowInvalid` (not `pa.ArrowTypeError`) when attempting to convert a non-null value to PyArrow's null type. This uncaught exception propagates to the user as a cryptic error.

2. **Inconsistent with pandas behavior**: Other pandas array types handle all-None arrays gracefully when calling fillna. The ArrowExtensionArray should provide consistent behavior or at least a clear error message.

3. **Legitimate use case**: Arrays containing only None values are valid and can occur naturally in data processing pipelines (e.g., as placeholders, during data initialization, or from filtering operations). The fillna method is specifically designed to handle missing values, so it should handle this edge case appropriately.

4. **Poor user experience**: The error message "Invalid null value" from PyArrow is cryptic and doesn't explain what went wrong or how to fix it. The intended error handler would provide: `"Invalid value '0' for dtype 'null[pyarrow]'"` which is much clearer.

5. **Documentation mismatch**: The fillna documentation doesn't indicate that it will crash on null-type arrays, and the existing error handling code shows clear intent to catch and handle type incompatibilities.

## Relevant Context

When PyArrow creates an array from only None values, it assigns the special "null" type to that array. The null type in PyArrow cannot hold any non-null values, which is why attempting to create a scalar with value 0 and type null raises an ArrowInvalid exception.

The code already has infrastructure in place to handle this scenario - it just needs to catch the correct exception type. The existing error handler at lines 1161-1163 converts PyArrow errors into user-friendly TypeErrors with clear messages about dtype incompatibility.

Relevant code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:1159-1163`

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1158,7 +1158,7 @@ class ArrowExtensionArray(

         try:
             fill_value = self._box_pa(value, pa_type=self._pa_array.type)
-        except pa.ArrowTypeError as err:
+        except (pa.ArrowTypeError, pa.ArrowInvalid) as err:
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
             raise TypeError(msg) from err
```