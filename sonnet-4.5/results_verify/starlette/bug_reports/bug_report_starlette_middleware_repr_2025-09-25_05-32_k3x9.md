# Bug Report: Middleware.__repr__ Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces malformed output with a leading comma when the middleware factory is a callable instance without a `__name__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware import Middleware


@given(
    args=st.lists(st.integers(), max_size=5),
    kwargs=st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=5)
)
def test_middleware_repr_no_leading_comma(args, kwargs):
    class CallableFactory:
        def __call__(self, app):
            return app

    factory = CallableFactory()

    middleware = Middleware(factory, *args, **kwargs)
    repr_str = repr(middleware)

    assert not repr_str.startswith("Middleware(,"), \
        f"repr has leading comma: {repr_str}"
```

**Failing input**: Any callable instance without `__name__` attribute

## Reproducing the Bug

```python
from starlette.middleware import Middleware


class CallableFactory:
    def __call__(self, app):
        return app


factory = CallableFactory()
middleware = Middleware(factory, "arg1", "arg2", key="value")

print(repr(middleware))
```

Output: `Middleware(, 'arg1', 'arg2', key='value')`

Expected: `Middleware('arg1', 'arg2', key='value')` or similar

## Why This Is A Bug

The issue occurs in `starlette/middleware/__init__.py:36-42`:

```python
def __repr__(self) -> str:
    class_name = self.__class__.__name__
    args_strings = [f"{value!r}" for value in self.args]
    option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
    name = getattr(self.cls, "__name__", "")
    args_repr = ", ".join([name] + args_strings + option_strings)
    return f"{class_name}({args_repr})"
```

When `self.cls` has no `__name__` attribute, `name` becomes an empty string. Line 41 then joins `[""] + args_strings + option_strings`, resulting in a leading comma in the output.

Callable instances (objects with `__call__` method) are valid middleware factories but don't have `__name__` by default, making this a legitimate edge case.

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -38,7 +38,10 @@ class Middleware:
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