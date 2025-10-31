# Bug Report: starlette.middleware.Middleware.__repr__ Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces invalid output with a leading comma when the middleware class's `__name__` attribute is an empty string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware import Middleware


@given(
    st.text(),
    st.lists(st.one_of(st.integers(), st.text(), st.booleans()), min_size=1),
    st.dictionaries(
        st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.one_of(st.integers(), st.text(), st.booleans()),
        min_size=0
    )
)
def test_middleware_repr_no_leading_comma(name, args, kwargs):
    class TestMiddleware:
        pass

    TestMiddleware.__name__ = name

    m = Middleware(TestMiddleware, *args, **kwargs)
    repr_str = repr(m)

    assert not repr_str.startswith("Middleware(, "), (
        f"Repr should not have leading comma, got: {repr_str}"
    )
```

**Failing input**: `name='', args=[0], kwargs={}`

## Reproducing the Bug

```python
from starlette.middleware import Middleware

class TestMiddleware:
    pass

TestMiddleware.__name__ = ""

m = Middleware(TestMiddleware, "arg1", "arg2", kwarg1="val1")
print(repr(m))
```

**Output**: `Middleware(, 'arg1', 'arg2', kwarg1='val1')`

**Expected**: `Middleware('arg1', 'arg2', kwarg1='val1')`

## Why This Is A Bug

The `__repr__` method should return a clean, valid string representation of the object. The current implementation produces output with a leading comma when the middleware class has an empty `__name__` attribute, which is syntactically invalid and violates the contract of `__repr__`.

## Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -36,7 +36,10 @@ class Middleware:
     def __repr__(self) -> str:
         class_name = self.__class__.__name__
         args_strings = [f"{value!r}" for value in self.args]
         option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
         name = getattr(self.cls, "__name__", "")
-        args_repr = ", ".join([name] + args_strings + option_strings)
+        parts = [name] if name else []
+        parts.extend(args_strings)
+        parts.extend(option_strings)
+        args_repr = ", ".join(parts)
         return f"{class_name}({args_repr})"
```