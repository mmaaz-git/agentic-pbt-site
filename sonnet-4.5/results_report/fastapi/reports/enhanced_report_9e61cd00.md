# Bug Report: fastapi.middleware.Middleware Invalid Python Syntax in __repr__ with Leading Comma

**Target**: `fastapi.middleware.Middleware` (via `starlette.middleware.Middleware`)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces syntactically invalid Python code with a leading comma when the middleware callable lacks a `__name__` attribute, violating Python's convention that `__repr__` should ideally return valid Python expressions.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovers the Middleware __repr__ bug."""

from hypothesis import given, strategies as st
from fastapi.middleware import Middleware


class CallableWithoutName:
    """A callable class that doesn't have a __name__ attribute."""
    def __call__(self, app, *args, **kwargs):
        return app


@given(args=st.lists(st.integers(), min_size=1, max_size=3))
def test_middleware_repr_with_unnamed_callable_and_args(args):
    """Test that Middleware.__repr__ doesn't produce invalid syntax with leading commas."""
    callable_without_name = CallableWithoutName()
    middleware = Middleware(callable_without_name, *args)
    repr_str = repr(middleware)

    assert not repr_str.startswith('Middleware(,'), \
        f"Found leading comma in repr: {repr_str}"


if __name__ == "__main__":
    # Run the test to find the bug
    test_middleware_repr_with_unnamed_callable_and_args()
```

<details>

<summary>
**Failing input**: `args=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 27, in <module>
    test_middleware_repr_with_unnamed_callable_and_args()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 15, in test_middleware_repr_with_unnamed_callable_and_args
    def test_middleware_repr_with_unnamed_callable_and_args(args):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 21, in test_middleware_repr_with_unnamed_callable_and_args
    assert not repr_str.startswith('Middleware(,'), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Found leading comma in repr: Middleware(, 0)
Falsifying example: test_middleware_repr_with_unnamed_callable_and_args(
    args=[0],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the Middleware __repr__ bug with leading comma."""

from fastapi.middleware import Middleware


class CallableWithoutName:
    """A callable class that doesn't have a __name__ attribute."""
    def __call__(self, app, *args, **kwargs):
        return app


# Create a Middleware instance with a callable lacking __name__
# and some additional arguments
middleware = Middleware(CallableWithoutName(), 123, foo="bar")

# Print the repr output
print("repr(middleware) output:")
print(repr(middleware))

# Also test without any args
middleware_no_args = Middleware(CallableWithoutName())
print("\nrepr(middleware_no_args) output:")
print(repr(middleware_no_args))

# Compare with a regular function that has __name__
def my_middleware(app):
    return app

middleware_with_name = Middleware(my_middleware, 456, bar="baz")
print("\nrepr(middleware_with_name) output:")
print(repr(middleware_with_name))

# Also test with a lambda (which has __name__ = '<lambda>')
middleware_lambda = Middleware(lambda app: app, 789)
print("\nrepr(middleware_lambda) output:")
print(repr(middleware_lambda))
```

<details>

<summary>
Invalid Python syntax with leading comma in repr output
</summary>
```
repr(middleware) output:
Middleware(, 123, foo='bar')

repr(middleware_no_args) output:
Middleware()

repr(middleware_with_name) output:
Middleware(my_middleware, 456, bar='baz')

repr(middleware_lambda) output:
Middleware(<lambda>, 789)
```
</details>

## Why This Is A Bug

This violates Python's fundamental convention for the `__repr__` method. According to the Python documentation (https://docs.python.org/3/reference/datamodel.html#object.__repr__), the `__repr__` method should "if at all possible, look like a valid Python expression that could be used to recreate an object with the same value."

The output `Middleware(, 123, foo='bar')` is syntactically invalid Python code that would cause a `SyntaxError` if evaluated. This occurs because:

1. When a callable lacks a `__name__` attribute (common with callable class instances), `getattr(self.cls, "__name__", "")` returns an empty string
2. The empty string is included as the first element in the list that gets joined: `[name] + args_strings + option_strings`
3. When joined with `", "`, this creates a leading comma: `", 123, foo='bar'"`

While the Starlette documentation doesn't explicitly specify the expected format of `__repr__`, producing syntactically invalid Python violates widely-accepted Python conventions and could break debugging tools that parse or evaluate repr output.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/__init__.py` at lines 40-41:

```python
def __repr__(self) -> str:
    class_name = self.__class__.__name__
    args_strings = [f"{value!r}" for value in self.args]
    option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
    name = getattr(self.cls, "__name__", "")  # Line 40: Gets empty string
    args_repr = ", ".join([name] + args_strings + option_strings)  # Line 41: Joins with empty string
    return f"{class_name}({args_repr})"
```

This affects:
- Callable class instances (classes with `__call__` method but no `__name__` attribute)
- Any custom callable object without a `__name__` attribute
- Does NOT affect regular functions, lambdas, or classes used as middleware

The bug only manifests when the callable has additional arguments or keyword arguments. Without arguments, `Middleware()` is produced, which is valid (though not very informative).

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