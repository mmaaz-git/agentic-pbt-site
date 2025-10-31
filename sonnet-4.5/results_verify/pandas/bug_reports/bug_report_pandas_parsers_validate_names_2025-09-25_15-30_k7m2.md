# Bug Report: pandas.io.parsers _validate_names TypeError

**Target**: `pandas.io.parsers.readers._validate_names`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_names` function raises `TypeError` instead of `ValueError` when given unhashable column names, violating its documented behavior and type contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import io

@given(
    unhashable_names=st.lists(
        st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=3), st.integers(), min_size=1, max_size=2)
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=50)
def test_unhashable_names_should_raise_valueerror(unhashable_names):
    csv_data = ','.join(['0'] * len(unhashable_names)) + '\n'
    try:
        df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
        assert False, "Should have raised an error"
    except TypeError:
        assert False, "Got TypeError but docstring promises ValueError"
    except ValueError:
        pass
```

**Failing input**: `unhashable_names = [[1], [2]]` or `unhashable_names = [{"a": 1}, {"b": 2}]`

## Reproducing the Bug

```python
import pandas as pd
import io

csv_data = "1,2,3\n4,5,6"
names = [[1, 2], [3, 4], [5, 6]]

try:
    df = pd.read_csv(io.StringIO(csv_data), names=names, header=None)
except TypeError as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")
    print("Expected: ValueError")
except ValueError as e:
    print(f"Correct: Raised {type(e).__name__}: {e}")
```

**Output**:
```
BUG: Raised TypeError: unhashable type: 'list'
Expected: ValueError
```

## Why This Is A Bug

The function `_validate_names` in `/pandas/io/parsers/readers.py` violates its contract in three ways:

1. **Documented behavior**: The docstring (line 561-562) states:
   > "Raise ValueError if the `names` parameter contains duplicates or has an invalid data type."

   It promises to raise `ValueError` for invalid data types, but raises `TypeError` instead.

2. **Type annotation**: The function signature (line 559) declares:
   ```python
   def _validate_names(names: Sequence[Hashable] | None) -> None:
   ```

   It expects `Sequence[Hashable]`, meaning validation should happen and raise `ValueError` for non-hashable elements.

3. **Exception hierarchy**: `TypeError` indicates wrong type, but `ValueError` indicates wrong value. Since the type is correct (it's a sequence), but the values are wrong (unhashable elements), `ValueError` is semantically correct.

The bug is on line 575:
```python
if len(names) != len(set(names)):
    raise ValueError("Duplicate names are not allowed.")
```

This calls `set(names)` before validating that elements are hashable, causing Python's `set()` to raise `TypeError` for unhashable elements.

## Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -572,9 +572,15 @@ def _validate_names(names: Sequence[Hashable] | None) -> None:
         If names are not unique or are not ordered (e.g. set).
     """
     if names is not None:
-        if len(names) != len(set(names)):
-            raise ValueError("Duplicate names are not allowed.")
         if not (
             is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
         ):
             raise ValueError("Names should be an ordered collection.")
+        try:
+            if len(names) != len(set(names)):
+                raise ValueError("Duplicate names are not allowed.")
+        except TypeError:
+            raise ValueError(
+                "Column names must be hashable. "
+                "Unhashable types like list, dict, or set are not allowed."
+            ) from None
```

This fix:
1. Checks collection type first (more specific validation)
2. Wraps the `set()` call in try-except to catch `TypeError`
3. Converts `TypeError` to `ValueError` with a clear message
4. Maintains backward compatibility for valid inputs