# Bug Report: result_type_many Incorrect Fallback for Mixed Dtypes

**Target**: `pandas.core.computation.common.result_type_many`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

In the `result_type_many` function, when handling a mix of extension array dtypes and many (>32) non-extension array dtypes, the ValueError fallback incorrectly uses `arrays_and_dtypes` instead of `non_ea_dtypes` in the reduce call. This defeats the purpose of separating extension array dtypes from non-extension array dtypes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.computation.common import result_type_many
from pandas import StringDtype
import numpy as np

@given(st.integers(min_value=40, max_value=50))
def test_result_type_many_mixed_dtypes(num_dtypes):
    non_ea_dtypes = [np.int32] * num_dtypes
    ea_dtype = StringDtype()
    all_dtypes = non_ea_dtypes + [ea_dtype]

    result = result_type_many(*all_dtypes)
    assert result is not None
```

**Failing input**: Any combination of >32 non-EA dtypes plus 1 or more EA dtypes (may not always fail, but the logic is incorrect)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.computation.common import result_type_many
from pandas import StringDtype

non_ea_dtypes = [np.int32] * 40
ea_dtype = StringDtype()
all_dtypes = non_ea_dtypes + [ea_dtype]

result = result_type_many(*all_dtypes)
```

Looking at the code, in the TypeError exception handler (line 29-48), the function:
1. Separates `arrays_and_dtypes` into `ea_dtypes` and `non_ea_dtypes` (lines 33-39)
2. When processing `non_ea_dtypes`, if there are too many (>32), it catches ValueError (line 44)
3. The fallback at line 45 incorrectly uses `arrays_and_dtypes` instead of `non_ea_dtypes`:

```python
except ValueError:
    np_dtype = reduce(np.result_type, arrays_and_dtypes)  # BUG: should be non_ea_dtypes
```

## Why This Is A Bug

The code path is designed to handle extension array dtypes separately from regular dtypes because `np.result_type` cannot handle extension array dtypes. The separation happens at lines 33-39.

When there are more than 32 non-EA dtypes, the code catches the ValueError and falls back to using `reduce`. However, line 45 incorrectly uses `arrays_and_dtypes` (which includes EA dtypes) instead of `non_ea_dtypes`.

This means:
1. The reduce will try to process extension array dtypes with `np.result_type`, which is exactly what the separation was meant to avoid
2. This could lead to errors or incorrect results
3. The logic is inconsistent with the intent of the code

## Fix

```diff
--- a/pandas/core/computation/common.py
+++ b/pandas/core/computation/common.py
@@ -42,7 +42,7 @@ def result_type_many(*arrays_and_dtypes):
         if non_ea_dtypes:
             try:
                 np_dtype = np.result_type(*non_ea_dtypes)
             except ValueError:
-                np_dtype = reduce(np.result_type, arrays_and_dtypes)
+                np_dtype = reduce(np.result_type, non_ea_dtypes)
             return find_common_type(ea_dtypes + [np_dtype])

         return find_common_type(ea_dtypes)
```