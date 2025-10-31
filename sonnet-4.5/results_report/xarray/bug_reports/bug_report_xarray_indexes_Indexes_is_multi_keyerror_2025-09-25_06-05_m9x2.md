# Bug Report: xarray.indexes Indexes.is_multi() KeyError on Invalid Key

**Target**: `xarray.core.indexes.Indexes.is_multi()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Indexes.is_multi()` method raises an unhelpful `KeyError` when called with a key that doesn't exist in the indexes, unlike the similar `get_all_coords()` method which has an `errors` parameter to handle this case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from xarray.core.indexes import PandasIndex, Indexes
from xarray.core.variable import Variable

@st.composite
def indexes_with_keys(draw):
    n_indexes = draw(st.integers(min_value=1, max_value=5))
    valid_keys = [f"dim_{i}" for i in range(n_indexes)]
    invalid_key = draw(st.text(min_size=1, max_size=10).filter(lambda x: x not in valid_keys))

    indexes_dict = {}
    variables_dict = {}
    for key in valid_keys:
        idx = PandasIndex(pd.Index([1, 2, 3]), key)
        var = Variable((key,), [1, 2, 3])
        indexes_dict[key] = idx
        variables_dict[key] = var

    return Indexes(indexes_dict, variables_dict), invalid_key

@settings(max_examples=100)
@given(indexes_with_keys())
def test_is_multi_invalid_key_error(indexes_and_key):
    indexes, invalid_key = indexes_and_key
    try:
        result = indexes.is_multi(invalid_key)
        assert False, "Expected an error for invalid key"
    except KeyError:
        pass
    except ValueError as e:
        assert "no index found" in str(e).lower()
```

**Failing input**: Any `Indexes` object with a key that doesn't exist, e.g., calling `indexes.is_multi("nonexistent")`

## Reproducing the Bug

```python
import pandas as pd
from xarray.core.indexes import PandasIndex, Indexes
from xarray.core.variable import Variable

idx1 = PandasIndex(pd.Index([1, 2, 3]), "x")
var1 = Variable(("x",), [1, 2, 3])

indexes_obj = Indexes({"x": idx1}, {"x": var1})

indexes_obj.is_multi("y")
```

**Output**:
```
Traceback (most recent call last):
  ...
  File "xarray/core/indexes.py", line 1775, in is_multi
    return len(self._id_coord_names[self._coord_name_id[key]]) > 1
KeyError: 'y'
```

## Why This Is A Bug

The API is inconsistent with the similar method `get_all_coords()`, which has an `errors` parameter to handle missing keys gracefully. The docstring for `is_multi()` doesn't mention that it will raise a `KeyError` for invalid keys, and the error message provides no context about what went wrong.

Users might reasonably expect one of the following behaviors:
1. Return `False` for a missing key (defensive)
2. Raise a `ValueError` with a clear message like `get_all_coords()` does
3. Have an `errors` parameter like `get_all_coords()`

Instead, they get a raw `KeyError` which doesn't explain that the key was not found in the indexes.

## Fix

Option 1: Check and raise informative error (minimal change)
```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
index abc123..def456 100644
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1771,6 +1771,9 @@ class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
     def is_multi(self, key: Hashable) -> bool:
         """Return True if ``key`` maps to a multi-coordinate index,
         False otherwise.
         """
+        if key not in self._indexes:
+            raise ValueError(f"no index found for {key!r} coordinate")
         return len(self._id_coord_names[self._coord_name_id[key]]) > 1
```

Option 2: Match `get_all_coords()` API with errors parameter (better consistency)
```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
index abc123..def456 100644
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1771,7 +1771,18 @@ class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
-    def is_multi(self, key: Hashable) -> bool:
+    def is_multi(self, key: Hashable, errors: ErrorOptions = "raise") -> bool:
         """Return True if ``key`` maps to a multi-coordinate index,
         False otherwise.
+
+        Parameters
+        ----------
+        key : hashable
+            Index key.
+        errors : {"raise", "ignore"}, default: "raise"
+            If "raise", raises a ValueError if `key` is not in indexes.
+            If "ignore", returns False instead.
         """
+        if key not in self._indexes:
+            if errors == "raise":
+                raise ValueError(f"no index found for {key!r} coordinate")
+            return False
         return len(self._id_coord_names[self._coord_name_id[key]]) > 1
```