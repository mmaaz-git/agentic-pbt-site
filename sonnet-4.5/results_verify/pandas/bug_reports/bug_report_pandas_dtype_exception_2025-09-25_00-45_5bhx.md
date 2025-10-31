# Bug Report: pandas_dtype Raises ValueError Instead of TypeError

**Target**: `pandas.api.types.pandas_dtype`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pandas_dtype` function's docstring states it "Raises TypeError if not a dtype", but for certain invalid dict inputs, it raises `ValueError` instead of `TypeError`, violating its documented contract.

## Property-Based Test

```python
from pandas.api.types import pandas_dtype
from hypothesis import given, strategies as st
import pytest


@given(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.text()),
    min_size=1,
    max_size=10
))
def test_pandas_dtype_raises_only_typeerror(d):
    """pandas_dtype should only raise TypeError for invalid inputs, per docstring."""
    try:
        pandas_dtype(d)
    except TypeError:
        pass
    except ValueError as e:
        pytest.fail(
            f"pandas_dtype raised ValueError instead of TypeError for input {d!r}.\n"
            f"Docstring says it should only raise TypeError.\n"
            f"Error was: {e}"
        )
```

**Failing input**: `{'0': ''}`

## Reproducing the Bug

```python
from pandas.api.types import pandas_dtype

invalid_input = {'0': ''}

try:
    pandas_dtype(invalid_input)
except ValueError as e:
    print(f"BUG: Raised ValueError: {e}")
    print("Expected: TypeError (per docstring)")
except TypeError as e:
    print(f"OK: Raised TypeError as documented: {e}")
```

Output:
```
BUG: Raised ValueError: entry not a 2- or 3- tuple
Expected: TypeError (per docstring)
```

## Why This Is A Bug

The function's docstring explicitly promises:

```
Raises
------
TypeError if not a dtype
```

However, when passed certain dict inputs (which are clearly not valid dtypes), the function raises `ValueError` instead of `TypeError`. This is a contract violation - the API documentation promises one exception type but delivers another.

## Fix

The issue is in `/pandas/core/dtypes/common.py` at line ~1663. The try-except block only catches `SyntaxError` and converts it to `TypeError`, but doesn't catch `ValueError` which can also be raised by `np.dtype()`.

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -1667,7 +1667,7 @@ def pandas_dtype(dtype) -> DtypeObj:
             warnings.simplefilter("always", DeprecationWarning)
             npdtype = np.dtype(dtype)
-    except SyntaxError as err:
+    except (SyntaxError, ValueError) as err:
         # np.dtype uses `eval` which can raise SyntaxError
         raise TypeError(f"data type '{dtype}' not understood") from err
```

This ensures all invalid inputs raise `TypeError` as documented, regardless of which specific error `np.dtype()` raises internally.