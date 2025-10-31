# Bug Report: django.db.models.functions.Substr Missing Length Validation

**Target**: `django.db.models.functions.Substr`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Substr` function accepts invalid `length` parameter values (zero and negative integers) without validation, while the similar `Left` function correctly validates that `length` must be greater than 0. This inconsistency violates the expected API contract and could lead to database errors.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from django.db.models.functions import Substr
from django.db.models.expressions import Value

@given(st.integers(max_value=0))
def test_property_substr_should_reject_non_positive_length(length):
    """Property: Substr should reject non-positive length values, like Left does."""
    with pytest.raises(ValueError):
        Substr(Value("test"), 1, length)
```

**Failing input**: Any non-positive integer for the `length` parameter, e.g., `length=0` or `length=-5`

## Reproducing the Bug

```python
from django.db.models.functions import Substr, Left
from django.db.models.expressions import Value

Left(Value("hello"), 0)
Left(Value("hello"), -5)

Substr(Value("hello"), 1, 0)
Substr(Value("hello"), 1, -5)
```

## Why This Is A Bug

The `Substr` and `Left` functions serve similar purposes (extracting characters from a string). The `Left` function validates that its `length` parameter must be greater than 0:

```python
if not hasattr(length, "resolve_expression"):
    if length < 1:
        raise ValueError("'length' must be greater than 0.")
```

However, `Substr` has no validation for its `length` parameter, accepting zero and negative values that would likely cause errors when the SQL query is executed. This API inconsistency violates user expectations and the principle of least surprise.

Similar functions in the same module (`LPad`, `Repeat`) also validate their numeric parameters to prevent invalid values. The lack of validation in `Substr` is an oversight.

## Fix

```diff
--- a/django/db/models/functions/text.py
+++ b/django/db/models/functions/text.py
@@ -355,6 +355,9 @@ class Substr(Func):
         if not hasattr(pos, "resolve_expression"):
             if pos < 1:
                 raise ValueError("'pos' must be greater than 0")
+        if length is not None and not hasattr(length, "resolve_expression"):
+            if length < 1:
+                raise ValueError("'length' must be greater than 0")
         expressions = [expression, pos]
         if length is not None:
             expressions.append(length)
```