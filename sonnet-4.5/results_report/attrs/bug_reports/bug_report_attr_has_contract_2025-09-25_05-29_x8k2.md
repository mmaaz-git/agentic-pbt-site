# Bug Report: attr.has() Contract Violation

**Target**: `attr.has`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attr.has()` function violates its documented API contract by accepting non-class inputs without raising `TypeError`, contradicting its explicit documentation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import attr
import inspect

@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
))
def test_has_should_raise_for_non_classes(non_class_value):
    """attr.has() should raise TypeError for non-class inputs per its documentation."""
    assume(not inspect.isclass(non_class_value))

    try:
        result = attr.has(non_class_value)
        raise AssertionError(
            f"attr.has({non_class_value!r}) returned {result} instead of raising TypeError"
        )
    except TypeError:
        pass
```

**Failing input**: `42` (or any non-class value like strings, lists, None, etc.)

## Reproducing the Bug

```python
import attr

result = attr.has(42)
```

The above code returns `False` instead of raising `TypeError` as documented.

```python
import attr

result = attr.has("not a class")
```

This also returns `False` instead of raising `TypeError`.

```python
import attr

result = attr.has(None)
```

This also returns `False` instead of raising `TypeError`.

## Why This Is A Bug

The `attr.has()` function's docstring explicitly states:

```python
def has(cls):
    """
    Check whether *cls* is a class with *attrs* attributes.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

    Returns:
        bool:
    """
```

The documentation clearly promises to raise `TypeError` if the input is not a class. However, the implementation (in `attr/_funcs.py` lines 326-351) does not validate that `cls` is actually a class. Instead, it just calls `getattr(cls, "__attrs_attrs__", None)` which succeeds on any object, including integers, strings, None, etc.

This means:
1. Users cannot rely on `TypeError` to catch programming errors where they pass wrong types
2. The function silently accepts invalid inputs instead of failing fast
3. Type checkers and documentation suggest only classes are valid, but runtime doesn't enforce this
4. **Inconsistency**: Other similar functions in attrs (`fields()` and `fields_dict()`) DO validate their inputs and raise `TypeError` for non-classes, making `has()` inconsistent with the rest of the API

## Fix

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -326,6 +326,9 @@ def astuple(
 def has(cls):
     """
     Check whether *cls* is a class with *attrs* attributes.

     Args:
         cls (type): Class to introspect.

     Raises:
         TypeError: If *cls* is not a class.

     Returns:
         bool:
     """
+    if not isinstance(cls, type):
+        raise TypeError(
+            f"cls must be a class, not {type(cls).__name__}"
+        )
+
     attrs = getattr(cls, "__attrs_attrs__", None)
     if attrs is not None:
         return True
```

Alternatively, if the intent is to accept any object (not just classes), the documentation should be updated to reflect this and the parameter should be renamed from `cls` to something like `obj`.