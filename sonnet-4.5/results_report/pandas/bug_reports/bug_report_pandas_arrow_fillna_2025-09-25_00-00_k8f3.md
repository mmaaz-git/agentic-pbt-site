# Bug Report: ArrowExtensionArray fillna raises ArrowInvalid on null-type arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `fillna()` on an `ArrowExtensionArray` containing only `None` values (which results in a PyArrow null type), the method raises an uncaught `pyarrow.lib.ArrowInvalid` exception instead of either handling the conversion gracefully or providing a clear error message.

## Property-Based Test

```python
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
```

**Failing input**: `values=[None]`, `fill_value=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
arr.fillna(0)
```

**Output:**
```
Traceback (most recent call last):
  File "...", line X, in ...
    result = arr.fillna(fill_value)
  File ".../pandas/core/arrays/arrow/array.py", line 1160, in fillna
    fill_value = self._box_pa(value, pa_type=self._pa_array.type)
  File ".../pandas/core/arrays/arrow/array.py", line 407, in _box_pa
    return cls._box_pa_scalar(value, pa_type)
  File ".../pandas/core/arrays/arrow/array.py", line 443, in _box_pa_scalar
    pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
pyarrow.lib.ArrowInvalid: Invalid null value
```

## Why This Is A Bug

1. **Incomplete exception handling**: The `fillna` method only catches `pa.ArrowTypeError` at line 1161, but PyArrow raises `pa.ArrowInvalid` when trying to convert a non-null value to the null type.

2. **Legitimate use case**: Users may reasonably create arrays with only `None` values (e.g., as placeholders) and expect `fillna` to work, especially since the method is designed to handle missing values.

3. **Inconsistent error handling**: The method has a try-except block specifically to catch type conversion errors and provide a user-friendly `TypeError`, but this case falls through, producing a confusing PyArrow-specific error.

4. **Poor user experience**: The error message "Invalid null value" is cryptic and doesn't explain what the user should do.

## Fix

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

This change ensures that `ArrowInvalid` exceptions (like the one raised when converting to a null type) are caught and converted to a more user-friendly `TypeError` with a clear message about the dtype incompatibility.