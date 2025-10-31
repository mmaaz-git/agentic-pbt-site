# Bug Report: starlette.middleware.Middleware.__repr__ Leading Comma with Empty Class Name

**Target**: `starlette.middleware.Middleware.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Middleware.__repr__` method produces syntactically invalid Python output with a leading comma when the middleware class's `__name__` attribute is an empty string.

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

if __name__ == "__main__":
    test_middleware_repr_no_leading_comma()
```

<details>

<summary>
**Failing input**: `name='', args=[0], kwargs={}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 28, in <module>
    test_middleware_repr_no_leading_comma()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 6, in test_middleware_repr_no_leading_comma
    st.text(),

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 23, in test_middleware_repr_no_leading_comma
    assert not repr_str.startswith("Middleware(, "), (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Repr should not have leading comma, got: Middleware(, 0)
Falsifying example: test_middleware_repr_no_leading_comma(
    # The test sometimes passed when commented parts were varied together.
    name='',
    args=[0],  # or any other generated value
    kwargs={},  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/50/hypo.py:24
```
</details>

## Reproducing the Bug

```python
from starlette.middleware import Middleware

class TestMiddleware:
    pass

TestMiddleware.__name__ = ""

m = Middleware(TestMiddleware, 0)
print(repr(m))
```

<details>

<summary>
Output shows invalid Python syntax with leading comma
</summary>
```
Middleware(, 0)
```
</details>

## Why This Is A Bug

The `__repr__` method is expected to return a clean, valid string representation of an object that, ideally, could be used to recreate that object. Python's conventions for `__repr__` specify that the output should be unambiguous and, when possible, should be valid Python code.

The current implementation in `starlette/middleware/__init__.py` (lines 36-42) unconditionally includes the middleware class name in the arguments list, even when it's empty. This results in output like `Middleware(, 0)` which is syntactically invalid Python - the leading comma creates a syntax error if someone tried to evaluate this string as Python code.

While classes with empty `__name__` attributes are uncommon, they are valid Python values, and the code should handle all valid inputs correctly. The bug violates the contract of `__repr__` by producing malformed output that doesn't follow Python's syntax rules.

## Relevant Context

The issue occurs in the `__repr__` method at line 41 of `/starlette/middleware/__init__.py`:

```python
def __repr__(self) -> str:
    class_name = self.__class__.__name__
    args_strings = [f"{value!r}" for value in self.args]
    option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
    name = getattr(self.cls, "__name__", "")
    args_repr = ", ".join([name] + args_strings + option_strings)  # Line 41
    return f"{class_name}({args_repr})"
```

The problem is that `[name]` always creates a list with one element, even when `name` is an empty string. When joined with commas, this empty string results in a leading comma.

This bug was discovered through property-based testing with Hypothesis, which systematically explores edge cases like empty strings that might be overlooked in manual testing.

## Proposed Fix

```diff
--- a/starlette/middleware/__init__.py
+++ b/starlette/middleware/__init__.py
@@ -38,7 +38,10 @@ class Middleware:
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