# Bug Report: Middleware.__repr__ Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces malformed output with a leading comma when the middleware factory doesn't have a `__name__` attribute (e.g., lambdas or callable objects).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from starlette.middleware import Middleware


@given(
    args=st.lists(st.one_of(st.integers(), st.text(max_size=20)), max_size=5),
    kwargs=st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))),
        st.one_of(st.integers(), st.text(max_size=20)),
        max_size=5
    ),
)
@settings(max_examples=1000)
def test_middleware_repr_no_leading_comma(args, kwargs):
    m = Middleware(lambda app: app, *args, **kwargs)
    repr_str = repr(m)

    assert not repr_str.startswith("Middleware(,"), \
        f"Middleware repr has leading comma: {repr_str}"
```

**Failing input**: Any lambda or callable without `__name__` attribute, e.g., `Middleware(lambda app: app, "arg1")`

## Reproducing the Bug

```python
from starlette.middleware import Middleware

m = Middleware(lambda app: app, "arg1", "arg2", key="value")
print(repr(m))
```

**Output**: `Middleware(, 'arg1', 'arg2', key='value')` (note the leading comma after the opening parenthesis)

**Expected**: `Middleware('arg1', 'arg2', key='value')` or `Middleware(<lambda>, 'arg1', 'arg2', key='value')`

## Why This Is A Bug

The `__repr__` method should produce a well-formed string representation. The current implementation unconditionally includes the middleware factory's `__name__` in the output, but when `__name__` doesn't exist (lambdas, callable objects), it defaults to an empty string, resulting in a leading comma: `Middleware(, ...)`.

This violates Python's convention that `repr()` should return a printable representation, ideally one that could be used to recreate the object.

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -36,8 +36,11 @@ class Middleware:
     def __repr__(self) -> str:
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