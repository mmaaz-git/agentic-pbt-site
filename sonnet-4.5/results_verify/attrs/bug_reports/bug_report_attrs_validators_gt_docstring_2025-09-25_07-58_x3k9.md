# Bug Report: attrs validators.gt Docstring Error

**Target**: `attr.validators.gt`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validators.gt()` function has incorrect documentation that claims it uses `operator.ge` when it actually uses `operator.gt`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr
from attr import validators
import operator

@given(st.integers())
def test_gt_uses_correct_operator(value):
    bound = 10
    v = validators.gt(bound)

    if value > bound:
        attr_obj = attr.Attribute(
            name="test", default=None, validator=None, repr=True,
            cmp=None, eq=True, eq_key=None, order=False,
            order_key=None, hash=None, init=True, kw_only=False,
            type=None, converter=None, metadata={}, alias=None
        )
        v(None, attr_obj, value)

    assert "operator.gt" in str(operator.gt)
    assert "operator.ge" in validators.gt.__doc__
```

**Failing input**: Any value triggers the docstring inconsistency.

## Reproducing the Bug

```python
import attr
from attr import validators

print(validators.gt.__doc__)
```

Output shows:
```
The validator uses `operator.ge` to compare the values.
```

But the actual implementation at line 495 in `attr/validators.py` is:
```python
return _NumberValidator(val, ">", operator.gt)
```

## Why This Is A Bug

The documentation explicitly states that `validators.gt` uses `operator.ge` for comparisons, but the implementation uses `operator.gt`. This violates the API contract documentation and could confuse users trying to understand the validator's behavior. The docstring was likely copy-pasted from the `ge` validator and not updated.

## Fix

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -485,7 +485,7 @@ def gt(val):
     A validator that raises `ValueError` if the initializer is called with a
     number smaller or equal to *val*.

-    The validator uses `operator.ge` to compare the values.
+    The validator uses `operator.gt` to compare the values.

     Args:
        val: Exclusive lower bound for values
```