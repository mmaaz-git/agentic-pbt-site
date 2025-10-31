# Bug Report: starlette.middleware Middleware.__repr__ crashes with None __name__

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method crashes with a `TypeError` when the wrapped middleware class has `__name__` set to `None` instead of a string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware import Middleware


class NoNameAttr:
    __name__ = None

    def __call__(self, app, *args, **kwargs):
        return app


def test_middleware_repr_with_class_without_name():
    no_name = NoNameAttr()
    middleware = Middleware(no_name, "arg1", key="value")
    repr_str = repr(middleware)
    assert isinstance(repr_str, str)
```

**Failing input**: A middleware class with `__name__ = None`

## Reproducing the Bug

```python
from starlette.middleware import Middleware


class MiddlewareWithNoneName:
    __name__ = None

    def __call__(self, app):
        return app


middleware = Middleware(MiddlewareWithNoneName())
repr(middleware)
```

Output:
```
TypeError: sequence item 0: expected str instance, NoneType found
```

## Why This Is A Bug

The `__repr__` method assumes that `getattr(self.cls, "__name__", "")` will always return a string. However, if a class explicitly sets `__name__ = None`, the `getattr` returns `None` (not the default `""`), which causes `str.join()` to fail.

While it's unusual for a class to have `__name__ = None`, this is valid Python and shouldn't crash the repr method.

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -38,7 +38,7 @@ class Middleware:
         class_name = self.__class__.__name__
         args_strings = [f"{value!r}" for value in self.args]
         option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
-        name = getattr(self.cls, "__name__", "")
+        name = getattr(self.cls, "__name__", "") or ""
         args_repr = ", ".join([name] + args_strings + option_strings)
         return f"{class_name}({args_repr})"
```

The fix ensures that even if `__name__` exists but is `None`, it will be converted to an empty string.