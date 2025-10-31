# Bug Report: Cython.Build.Inline.safe_type AttributeError with Custom Classes and None Context

**Target**: `Cython.Build.Inline.safe_type`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `safe_type` function crashes with an `AttributeError` when processing custom class instances with its default `context=None` parameter, attempting to call `find_module()` on None.

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

# Run the test
test_safe_type_with_custom_class_no_context()
```

<details>

<summary>
**Failing input**: `CustomClass()`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 13, in <module>
    test_safe_type_with_custom_class_no_context()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 8, in test_safe_type_with_custom_class_no_context
    def test_safe_type_with_custom_class_no_context(obj):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 9, in test_safe_type_with_custom_class_no_context
    result = safe_type(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Inline.py", line 89, in safe_type
    module = context.find_module(base_type.__module__, need_pxd=False)
             ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'find_module'
Falsifying example: test_safe_type_with_custom_class_no_context(
    obj=CustomClass(),
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Inline import safe_type

class CustomClass:
    pass

obj = CustomClass()
result = safe_type(obj)
print(f"Result: {result}")
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'find_module'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/repo.py", line 7, in <module>
    result = safe_type(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Inline.py", line 89, in safe_type
    module = context.find_module(base_type.__module__, need_pxd=False)
             ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'find_module'
```
</details>

## Why This Is A Bug

The `safe_type` function signature explicitly defines `context=None` as the default parameter value, establishing that the function should work without a context. This is a public API function (no underscore prefix) that can be imported directly: `from Cython.Build.Inline import safe_type`.

The function correctly handles built-in types (int, float, bool, list, tuple, dict, str, complex) with `context=None`, returning appropriate type strings. However, when processing custom class instances, the function traverses the Method Resolution Order (MRO) and encounters the custom class before reaching the built-in `object` base class. At line 89, it attempts to call `context.find_module()` without checking if `context` is None, causing an AttributeError crash.

The expected behavior when `context=None` is to gracefully degrade and return `'object'` for custom classes, just as it does when no matching Cython type is found. The function should not crash with an unhandled exception when using its default parameter value.

## Relevant Context

The `safe_type` function is used internally by `unsafe_type` (line 70) and indirectly by `cython_inline` (line 170, through the `get_type` parameter). In these internal uses, a context is typically provided, which explains why this bug might not have been discovered earlier.

The bug occurs specifically in the MRO traversal logic (lines 86-94 of `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Inline.py`):

1. For a custom class, the MRO is `[CustomClass, object]`
2. The first iteration processes `CustomClass` with `__module__='__main__'`
3. Since `'__main__'` is not in `('__builtin__', 'builtins')`, the code continues to line 89
4. Line 89 attempts to call `context.find_module()` with `context=None`, causing the crash
5. The function never reaches the `object` base class that would return `'object'`

## Proposed Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -86,7 +86,10 @@ def safe_type(arg, context=None):
         for base_type in py_type.__mro__:
             if base_type.__module__ in ('__builtin__', 'builtins'):
                 return 'object'
-            module = context.find_module(base_type.__module__, need_pxd=False)
+            if context is not None:
+                module = context.find_module(base_type.__module__, need_pxd=False)
+            else:
+                module = None
             if module:
                 entry = module.lookup(base_type.__name__)
                 if entry.is_type:
```