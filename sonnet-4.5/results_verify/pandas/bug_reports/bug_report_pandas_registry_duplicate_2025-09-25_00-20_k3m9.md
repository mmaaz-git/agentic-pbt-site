# Bug Report: pandas ExtensionDtype Registry Allows Unbounded Duplicate Registrations

**Target**: `pandas.api.extensions.register_extension_dtype` / `pandas.core.dtypes.base.Registry.register`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The ExtensionDtype registry allows the same dtype class to be registered multiple times, causing unbounded memory growth and silently ignoring subsequent registrations during lookups.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import register_extension_dtype, ExtensionDtype
from pandas.core.dtypes.base import _registry


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=50)
def test_registry_grows_unbounded_with_duplicate_registrations(n):
    dtype_name = f"test_dtype_{n}"
    initial_count = len(_registry.dtypes)

    class MyDtype(ExtensionDtype):
        name = dtype_name
        type = int

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import ExtensionArray
            class MyArray(ExtensionArray):
                dtype = cls()
                def __init__(self, data):
                    self._data = np.array(data)
                def __len__(self):
                    return len(self._data)
                def __getitem__(self, item):
                    return self._data[item]
                def isna(self):
                    return np.zeros(len(self._data), dtype=bool)
            return MyArray

    for i in range(n):
        register_extension_dtype(MyDtype)

    final_count = len(_registry.dtypes)
    growth = final_count - initial_count

    assert growth == n
```

**Failing input**: Registering the same dtype class multiple times (e.g., 10 times)

## Reproducing the Bug

```python
import numpy as np
from pandas.api.extensions import register_extension_dtype, ExtensionDtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import _registry


class MyDtype(ExtensionDtype):
    name = "mydtype"
    type = int

    @classmethod
    def construct_array_type(cls):
        class MyArray(ExtensionArray):
            dtype = cls()

            def __init__(self, data):
                self._data = np.array(data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, item):
                return self._data[item]

            def isna(self):
                return np.zeros(len(self._data), dtype=bool)

        return MyArray


initial = len(_registry.dtypes)
register_extension_dtype(MyDtype)
register_extension_dtype(MyDtype)
register_extension_dtype(MyDtype)
final = len(_registry.dtypes)

print(f"Registry grew by {final - initial} (expected 1, got {final - initial})")
```

## Why This Is A Bug

The `Registry.register` method unconditionally appends dtypes to its list without checking for duplicates. This violates the expected behavior of a registration system, which should be idempotent. This causes:

1. **Memory leak**: In module reload scenarios or accidental re-registration, the registry grows unbounded
2. **Silent failures**: Registering a different class with the same name silently adds it but lookups only find the first one
3. **Performance degradation**: The `find` method must iterate through all duplicates

## Fix

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -520,6 +520,12 @@ class Registry:
     def register(self, dtype: type_t[ExtensionDtype]) -> None:
         """
         Parameters
         ----------
         dtype : ExtensionDtype class
         """
         if not issubclass(dtype, ExtensionDtype):
             raise ValueError("can only register pandas extension dtypes")

+        if dtype in self.dtypes:
+            return
+
         self.dtypes.append(dtype)
```