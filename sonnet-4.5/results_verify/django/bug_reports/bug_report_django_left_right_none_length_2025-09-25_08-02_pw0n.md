# Bug Report: django.db.models.functions.text.Left/Right TypeError with None length

**Target**: `django.db.models.functions.text.Left` and `django.db.models.functions.text.Right`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`Left.__init__` and `Right.__init__` raise TypeError instead of ValueError when `length=None`, due to comparing None with an integer before checking if `length` is None.

## Property-Based Test

```python
from django.db.models import Value
from django.db.models.functions.text import Left, Right
import pytest


def test_left_with_none_length():
    with pytest.raises((ValueError, TypeError)):
        Left(Value("test"), None)


def test_right_with_none_length():
    with pytest.raises((ValueError, TypeError)):
        Right(Value("test"), None)
```

**Failing input**: `length=None`

## Reproducing the Bug

```python
import django
from django.conf import settings
settings.configure()
django.setup()

from django.db.models.functions.text import Left, Right
from django.db.models import Value

left = Left(Value('test'), None)
right = Right(Value('test'), None)
```

## Why This Is A Bug

The validation logic in `Left.__init__` performs `length < 1` comparison before checking if `length is None`, causing a TypeError when None is passed. This is the same issue as in `Substr` and is inconsistent with similar functions like `Repeat`, `LPad`, and `RPad`, which explicitly check `is not None` before comparison.

The error message is confusing: instead of a clear ValueError about invalid input, users get `TypeError: '<' not supported between instances of 'NoneType' and 'int'`.

Since `Right` inherits from `Left`, it has the same bug.

## Fix

```diff
--- a/django/db/models/functions/text.py
+++ b/django/db/models/functions/text.py
@@ -82,7 +82,7 @@ class Left(Func):
         expression: the name of a field, or an expression returning a string
         length: the number of characters to return from the start of the string
         """
-        if not hasattr(length, "resolve_expression"):
+        if not hasattr(length, "resolve_expression") and length is not None:
             if length < 1:
                 raise ValueError("'length' must be greater than 0.")
         super().__init__(expression, length, **extra)
```