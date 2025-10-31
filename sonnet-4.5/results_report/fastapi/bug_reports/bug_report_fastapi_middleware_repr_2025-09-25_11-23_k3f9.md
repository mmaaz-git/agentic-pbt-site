# Bug Report: fastapi.middleware.Middleware Leading Comma in __repr__

**Target**: `fastapi.middleware.Middleware` (via `starlette.middleware.Middleware`)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces invalid Python syntax with a leading comma when the middleware callable lacks a `__name__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


@given(args=st.lists(st.integers(), min_size=1, max_size=3))
def test_middleware_repr_with_unnamed_callable_and_args(args):
    callable_without_name = CallableWithoutName()
    middleware = Middleware(callable_without_name, *args)
    repr_str = repr(middleware)

    assert not repr_str.startswith('Middleware(,'), \
        f"Found leading comma in repr: {repr_str}"
```

**Failing input**: `args=[0]` (or any args/kwargs)

## Reproducing the Bug

```python
from fastapi.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


middleware = Middleware(CallableWithoutName(), 123, foo="bar")
print(repr(middleware))
```

**Output**: `Middleware(, 123, foo='bar')`
**Expected**: `Middleware(123, foo='bar')` or `Middleware(<callable>, 123, foo='bar')`

## Why This Is A Bug

The `__repr__` method violates Python conventions by producing syntactically invalid output. The leading comma occurs when the callable has no `__name__` attribute (e.g., callable class instances). This happens because:

1. Line 40 in `starlette/middleware/__init__.py`: `name = getattr(self.cls, "__name__", "")` gets an empty string
2. Line 41: `args_repr = ", ".join([name] + args_strings + option_strings)` joins with empty string first
3. Result: `Middleware(, arg1, arg2)` instead of `Middleware(arg1, arg2)`

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -37,7 +37,10 @@ class Middleware:
         class_name = self.__class__.__name__
         args_strings = [f"{value!r}" for value in self.args]
         option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
         name = getattr(self.cls, "__name__", "")
-        args_repr = ", ".join([name] + args_strings + option_strings)
+        if name:
+            args_repr = ", ".join([name] + args_strings + option_strings)
+        else:
+            args_repr = ", ".join(args_strings + option_strings)
         return f"{class_name}({args_repr})"
```