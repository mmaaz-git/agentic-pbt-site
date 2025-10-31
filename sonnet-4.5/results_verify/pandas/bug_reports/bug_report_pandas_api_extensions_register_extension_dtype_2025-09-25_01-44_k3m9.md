# Bug Report: pandas.api.extensions.register_extension_dtype Duplicate Registration

**Target**: `pandas.api.extensions.register_extension_dtype`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `register_extension_dtype` is called multiple times with different classes that have the same `name` attribute, the registry does not override the first registration. Instead, it appends the new dtype to the list, but the registry's `find` method always returns the first registered dtype with that name, making subsequent registrations silently ineffective.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd
from pandas.api.extensions import register_extension_dtype, ExtensionDtype


@given(dtype_name=st.text(alphabet=st.characters(whitelist_categories=('L',), min_codepoint=97, max_codepoint=122), min_size=1, max_size=30))
def test_register_extension_dtype_idempotence(dtype_name):
    assume(dtype_name.isidentifier())

    @register_extension_dtype
    class TestDtype1(ExtensionDtype):
        name = dtype_name
        type = object
        _metadata = ()

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import PandasArray
            return PandasArray

    @register_extension_dtype
    class TestDtype2(ExtensionDtype):
        name = dtype_name
        type = object
        _metadata = ()

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import PandasArray
            return PandasArray

    dtype = pd.api.types.pandas_dtype(dtype_name)
    assert isinstance(dtype, TestDtype2)
```

**Failing input**: `dtype_name='a'` (or any valid identifier)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.extensions import register_extension_dtype, ExtensionDtype

@register_extension_dtype
class FirstDtype(ExtensionDtype):
    name = 'duplicatename'
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

@register_extension_dtype
class SecondDtype(ExtensionDtype):
    name = 'duplicatename'
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

retrieved_dtype = pd.api.types.pandas_dtype('duplicatename')
assert isinstance(retrieved_dtype, FirstDtype)
assert not isinstance(retrieved_dtype, SecondDtype)
```

## Why This Is A Bug

The `register_extension_dtype` function allows multiple dtypes with the same name to be registered, but only the first one can ever be retrieved. This is inconsistent with the similar `register_series_accessor` function, which warns when overriding a preexisting attribute and does perform the override.

Users might reasonably expect that re-registering a dtype with the same name would update the registration, similar to how dictionary assignment works. The silent failure to override creates confusion and makes it impossible to programmatically update a dtype registration.

## Fix

The `register` method in the `Registry` class should either:
1. Warn and override when a duplicate name is detected (consistent with accessor registration)
2. Raise an error to prevent duplicate names
3. Check for duplicates and remove the old entry before appending

Here's a patch implementing option 1:

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -1580,6 +1580,16 @@ class Registry:
         """
         if not issubclass(dtype, ExtensionDtype):
             raise ValueError("can only register pandas extension dtypes")
+
+        # Check for existing dtype with same name
+        for i, existing_dtype in enumerate(self.dtypes):
+            if existing_dtype.name == dtype.name:
+                warnings.warn(
+                    f"registration of extension dtype {repr(dtype)} under name "
+                    f"{repr(dtype.name)} is overriding a preexisting "
+                    f"dtype {repr(existing_dtype)} with the same name.",
+                    UserWarning,
+                    stacklevel=find_stack_level(),
+                )
+                self.dtypes[i] = dtype
+                return

         self.dtypes.append(dtype)
```