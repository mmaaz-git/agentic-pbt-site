# Bug Report: starlette.middleware.Middleware.__repr__ Invalid Syntax with Leading Comma

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces syntactically invalid Python code containing a leading comma when the middleware callable lacks a `__name__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


@given(
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=3)
)
def test_middleware_repr_no_leading_comma(args, kwargs):
    callable_obj = CallableWithoutName()
    middleware = Middleware(callable_obj, *args, **kwargs)
    repr_str = repr(middleware)

    if args or kwargs:
        assert not repr_str.split("(", 1)[1].startswith(", "), \
            f"repr should not have leading comma: {repr_str}"


# Run the test with the specific failing input
if __name__ == "__main__":
    # Call the underlying function directly
    callable_obj = CallableWithoutName()
    middleware = Middleware(callable_obj, 0)
    repr_str = repr(middleware)
    print(f"Test with args=[0], kwargs={{}}")
    print(f"Result: {repr_str}")
    if [0] or {}:
        if repr_str.split("(", 1)[1].startswith(", "):
            print(f"FAILED: repr should not have leading comma: {repr_str}")
        else:
            print("PASSED")
```

<details>

<summary>
**Failing input**: `args=[0], kwargs={}`
</summary>
```
Test with args=[0], kwargs={}
Result: Middleware(, 0)
FAILED: repr should not have leading comma: Middleware(, 0)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


# Create an instance of the callable without __name__ attribute
callable_obj = CallableWithoutName()

# Create Middleware with this callable and some arguments
middleware = Middleware(callable_obj, 123, kwarg="test")

# Print the repr to demonstrate the bug
print(repr(middleware))
```

<details>

<summary>
Output shows invalid syntax with leading comma
</summary>
```
Middleware(, 123, kwarg='test')
```
</details>

## Why This Is A Bug

This violates Python's well-established convention that `__repr__` should produce syntactically valid Python code. The current implementation in `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/__init__.py:36-42` concatenates an empty string when `self.cls` lacks a `__name__` attribute (line 40: `name = getattr(self.cls, "__name__", "")`). This empty string becomes the first element in the list passed to `", ".join()` on line 41, resulting in a leading comma that creates syntactically invalid Python: `Middleware(, 123, kwarg='test')`.

Python's documentation and community conventions establish that repr output should be "a string containing a printable representation of an object" that ideally can be used to recreate the object via `eval()`. While not all reprs can be eval-able, they should at minimum be syntactically valid Python expressions. The output `Middleware(, 123, kwarg='test')` would raise a `SyntaxError` if evaluated, violating this fundamental expectation.

## Relevant Context

This bug affects real-world scenarios where developers use:
- Callable class instances as middleware factories (common pattern for stateful middleware)
- Custom callable objects without `__name__` attributes
- Any callable that doesn't define `__name__` (though lambdas do have `__name__` set to `"<lambda>"`)

The bug is located in the starlette middleware module at line 41 where the join operation unconditionally includes the name (even when empty):
- File: `starlette/middleware/__init__.py`
- Method: `Middleware.__repr__`
- Lines: 36-42

Python conventions for `__repr__`:
- [Python Data Model Documentation](https://docs.python.org/3/reference/datamodel.html#object.__repr__)
- PEP 8 recommends repr be unambiguous and, when sensible, match valid Python syntax

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

     def __iter__(self) -> Iterator[Any]:
```