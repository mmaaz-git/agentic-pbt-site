# Bug Report: django.core.checks.CheckMessage Equality Not Symmetric

**Target**: `django.core.checks.CheckMessage.__eq__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CheckMessage.__eq__` method violates the symmetry property of equality when comparing instances of the parent class with instances of subclasses (e.g., `Error`, `Warning`). Specifically, `CheckMessage(ERROR, msg) == Error(msg)` returns `True`, but `Error(msg) == CheckMessage(ERROR, msg)` returns `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks import CheckMessage, Error, ERROR


@given(st.text())
def test_checkmessage_equality_symmetry_with_subclass(msg):
    parent = CheckMessage(ERROR, msg)
    child = Error(msg)

    assert (parent == child) == (child == parent), (
        f"Symmetry violated: CheckMessage == Error is {parent == child}, "
        f"but Error == CheckMessage is {child == parent}"
    )
```

**Failing input**: Any string, e.g., `msg = "test"`

## Reproducing the Bug

```python
from django.core.checks import CheckMessage, Error, ERROR

parent = CheckMessage(ERROR, "Test message")
child = Error("Test message")

print("parent == child:", parent == child)
print("child == parent:", child == parent)

assert (parent == child) == (child == parent)
```

Output:
```
parent == child: True
child == parent: False
AssertionError
```

## Why This Is A Bug

The equality operator should be symmetric: if `a == b`, then `b == a` must also be true. This is a fundamental property of equality in mathematics and is expected by Python developers. The current implementation uses `isinstance(other, self.__class__)`, which causes asymmetric behavior when comparing parent and child class instances.

This bug could lead to unexpected behavior in code that relies on equality being symmetric, such as:
- Using `CheckMessage` objects as dictionary keys
- Comparing collections containing mixed `CheckMessage` and subclass instances
- Any code that assumes `a == b` implies `b == a`

## Fix

Replace the `isinstance(other, self.__class__)` check with a check that allows comparison between instances of the same class hierarchy while maintaining symmetry:

```diff
--- a/django/core/checks/messages.py
+++ b/django/core/checks/messages.py
@@ -18,7 +18,7 @@ class CheckMessage:

     def __eq__(self, other):
-        return isinstance(other, self.__class__) and all(
+        return isinstance(other, CheckMessage) and all(
             getattr(self, attr) == getattr(other, attr)
             for attr in ["level", "msg", "hint", "obj", "id"]
         )
```

This change ensures that any two `CheckMessage` instances (including subclasses) are compared based on their attributes rather than their exact class type, maintaining symmetry while still preventing comparison with unrelated types.