# Bug Report: Middleware.__repr__ Produces Invalid Python Syntax with Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method generates syntactically invalid Python code with a leading comma when used with callable instances that lack a `__name__` attribute.

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


# Run the test
test_middleware_repr_no_leading_comma()
```

<details>

<summary>
**Failing input**: `args=[0], kwargs={}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 24, in <module>
    test_middleware_repr_no_leading_comma()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 6, in test_middleware_repr_no_leading_comma
    args=st.lists(st.integers(), max_size=5),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 19, in test_middleware_repr_no_leading_comma
    assert not repr_str.startswith("Middleware(,"), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: repr has leading comma: Middleware(, 0)
Falsifying example: test_middleware_repr_no_leading_comma(
    args=[0],
    kwargs={},
)
```
</details>

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

<details>

<summary>
Output shows invalid Python syntax with leading comma
</summary>
```
Middleware(, 'arg1', 'arg2', key='value')
```
</details>

## Why This Is A Bug

This violates Python's documented `__repr__` convention that the output should either be "a valid Python expression that could be used to recreate an object" or a descriptive string in angle brackets. The output `Middleware(, 'arg1', 'arg2', key='value')` is syntactically invalid Python code due to the leading comma after the opening parenthesis.

The bug occurs because when a callable instance (an object with a `__call__` method) is used as a middleware factory without having a `__name__` attribute, the code at line 40 of `/starlette/middleware/__init__.py` retrieves an empty string. Line 41 then joins this empty string with the arguments list, resulting in `["", "arg1", "arg2"]` being joined with `", "`, which produces `", arg1, arg2"`.

Callable instances are valid middleware factories according to Starlette's type hints (the `_MiddlewareFactory` protocol only requires a callable with the correct signature). This makes it a legitimate use case that should be handled correctly.

## Relevant Context

- Python's official documentation states that `__repr__` should return valid Python syntax or a descriptive string (https://docs.python.org/3/reference/datamodel.html#object.__repr__)
- The Starlette type system explicitly supports callable instances via the `_MiddlewareFactory` protocol (line 17-18 in the source)
- Functions and classes with `__name__` attributes work correctly, producing output like `Middleware(function_name, 'arg1')`
- The issue only manifests when there are additional arguments; `Middleware()` with no args works fine even with callable instances

## Proposed Fix

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