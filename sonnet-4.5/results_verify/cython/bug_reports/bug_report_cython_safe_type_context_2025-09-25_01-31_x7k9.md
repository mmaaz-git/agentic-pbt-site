# Bug Report: Cython.Build.Inline.safe_type AttributeError with None Context

**Target**: `Cython.Build.Inline.safe_type`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `safe_type` crashes with `AttributeError` when called with a custom class instance and `context=None` (the default), because it attempts to call `context.find_module()` without checking if context is None.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Inline import safe_type

class CustomClass:
    pass

@given(st.builds(CustomClass))
def test_safe_type_with_custom_class_no_context(obj):
    result = safe_type(obj)
    assert isinstance(result, str)
```

**Failing input**: Any instance of a user-defined class

## Reproducing the Bug

```python
from Cython.Build.Inline import safe_type

class CustomClass:
    pass

obj = CustomClass()
result = safe_type(obj)
```

This crashes with:
```
AttributeError: 'NoneType' object has no attribute 'find_module'
```

## Why This Is A Bug

The function signature allows `context=None` as the default:

```python
def safe_type(arg, context=None):
```

However, for custom class instances, the code reaches line 89:

```python
module = context.find_module(base_type.__module__, need_pxd=False)
```

When `context` is `None`, this raises `AttributeError`.

The issue occurs because the MRO iteration processes custom classes BEFORE reaching the builtin `object` base class:
1. For `CustomClass`, MRO = `[CustomClass, object]`
2. First iteration: `base_type=CustomClass`, `base_type.__module__='__main__'` (not builtins)
3. Code tries to call `context.find_module('__main__', ...)` with `context=None`
4. Crashes before reaching `object` in the MRO

## Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -86,7 +86,7 @@ def safe_type(arg, context=None):
         for base_type in py_type.__mro__:
             if base_type.__module__ in ('__builtin__', 'builtins'):
                 return 'object'
-            module = context.find_module(base_type.__module__, need_pxd=False)
+            module = context.find_module(base_type.__module__, need_pxd=False) if context else None
             if module:
                 entry = module.lookup(base_type.__name__)
                 if entry.is_type:
```