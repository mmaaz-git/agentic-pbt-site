# Bug Report: starlette.middleware.Middleware.__repr__ Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces invalid Python-like output with a leading comma when the wrapped callable has no `__name__` attribute, resulting in output like `Middleware(, arg1, arg2)` instead of the expected `Middleware(arg1, arg2)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware import Middleware


@given(
    args=st.lists(st.integers(), min_size=1, max_size=5)
)
@settings(max_examples=100)
def test_middleware_repr_no_leading_comma(args):
    class CallableWithoutName:
        def __call__(self, app, *args, **kwargs):
            return app

    obj = CallableWithoutName()
    m = Middleware(obj, *args)
    repr_str = repr(m)

    assert not repr_str.startswith("Middleware(, "), \
        f"Found leading comma in repr: {repr_str}"
```

**Failing input**: Any callable without `__name__` attribute and with at least one argument

## Reproducing the Bug

```python
from starlette.middleware import Middleware

class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app

obj = CallableWithoutName()
m = Middleware(obj, 1, 2, x=3)
print(repr(m))
```

**Output**: `Middleware(, 1, 2, x=3)`

**Expected**: `Middleware(1, 2, x=3)` or `Middleware(CallableWithoutName, 1, 2, x=3)`

## Why This Is A Bug

The `__repr__` method is documented to return a "string representation" of the object that should ideally be valid Python code that could recreate the object. When a callable has no `__name__` attribute, the current implementation uses an empty string as the name and joins it with args, producing invalid syntax with a leading comma.

This violates the API contract for `__repr__` which should produce valid, readable output. While this is cosmetic and doesn't affect functionality, it makes debugging harder.

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -36,9 +36,11 @@ class Middleware:
     def __repr__(self) -> str:
         class_name = self.__class__.__name__
         args_strings = [f"{value!r}" for value in self.args]
         option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
         name = getattr(self.cls, "__name__", "")
-        args_repr = ", ".join([name] + args_strings + option_strings)
+        name_list = [name] if name else []
+        args_repr = ", ".join(name_list + args_strings + option_strings)
         return f"{class_name}({args_repr})"
```