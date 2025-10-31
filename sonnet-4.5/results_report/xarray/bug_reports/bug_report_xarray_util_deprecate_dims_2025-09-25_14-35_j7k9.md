# Bug Report: xarray.util.deprecation_helpers.deprecate_dims Parameter Override

**Target**: `xarray.util.deprecation_helpers.deprecate_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deprecate_dims` decorator incorrectly overwrites an explicitly provided `dim` parameter when both the deprecated `dims` parameter and the new `dim` parameter are provided. The deprecated parameter should not take precedence over the new one.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import deprecate_dims


@given(
    dims_value=st.text(),
    dim_value=st.text()
)
def test_deprecate_dims_precedence(dims_value, dim_value):
    @deprecate_dims
    def func(*, dim=None):
        return dim

    result = func(dims=dims_value, dim=dim_value)

    assert result == dim_value, \
        f"Expected dim={dim_value!r} to take precedence, but got {result!r}"
```

**Failing input**: Any input where `dims_value != dim_value`, e.g., `dims_value="x"`, `dim_value="y"`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import deprecate_dims


@deprecate_dims
def example_func(*, dim=None):
    return dim


result = example_func(dims="x", dim="y")
print(f"Expected: 'y'")
print(f"Got: {result!r}")
```

Output:
```
Expected: 'y'
Got: 'x'
```

## Why This Is A Bug

When a user explicitly provides both the deprecated parameter (`dims`) and the new parameter (`dim`), the new parameter should take precedence. The current implementation does the opposite:

```python
def deprecate_dims(func: T, old_name="dims") -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if old_name in kwargs:
            emit_user_level_warning(...)
            kwargs["dim"] = kwargs.pop(old_name)  # Overwrites kwargs["dim"]!
        return func(*args, **kwargs)
    return wrapper
```

This can cause unexpected behavior for users who are in the process of migrating from `dims` to `dim`. If they update some calls to use `dim` but accidentally leave `dims` in place, the old value will silently override their intended value.

## Fix

The decorator should check if `dim` is already in kwargs and either raise an error or skip the deprecation warning and use the new parameter:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -133,6 +133,11 @@ def deprecate_dims(func: T, old_name="dims") -> T:
     @wraps(func)
     def wrapper(*args, **kwargs):
         if old_name in kwargs:
+            if "dim" in kwargs:
+                raise TypeError(
+                    f"Cannot specify both {old_name!r} and 'dim'. "
+                    f"Please use 'dim' only."
+                )
             emit_user_level_warning(
                 f"The `{old_name}` argument has been renamed to `dim`, and will be removed "
                 "in the future. This renaming is taking place throughout xarray over the "
```

Alternatively, for a gentler approach, prioritize the new parameter:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -133,7 +133,9 @@ def deprecate_dims(func: T, old_name="dims") -> T:
     @wraps(func)
     def wrapper(*args, **kwargs):
         if old_name in kwargs:
+            if "dim" not in kwargs:
+                kwargs["dim"] = kwargs.pop(old_name)
-            emit_user_level_warning(
+                emit_user_level_warning(
                 f"The `{old_name}` argument has been renamed to `dim`, and will be removed "
                 "in the future. This renaming is taking place throughout xarray over the "
                 "next few releases.",
@@ -141,8 +143,9 @@ def deprecate_dims(func: T, old_name="dims") -> T:
                 PendingDeprecationWarning,
             )
-            kwargs["dim"] = kwargs.pop(old_name)
+            else:
+                kwargs.pop(old_name)  # Remove deprecated param, keep new one
         return func(*args, **kwargs)
```