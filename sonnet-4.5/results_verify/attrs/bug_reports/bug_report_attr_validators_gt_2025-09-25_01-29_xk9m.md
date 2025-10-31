# Bug Report: attr.validators.gt() Documentation Error

**Target**: `attr.validators.gt()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `attr.validators.gt()` incorrectly states that it uses `operator.ge` for comparisons, when it actually uses `operator.gt`. This creates a mismatch between the documented and actual behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

@given(st.integers(min_value=-100, max_value=100))
def test_gt_validator_uses_correct_operator(value):
    bound = 50

    @attr.define
    class TestClass:
        x: int = attr.field(validator=attr.validators.gt(bound))

    if value > bound:
        obj = TestClass(x=value)
        assert obj.x == value
    else:
        try:
            TestClass(x=value)
            assert False, f"Should have rejected {value} <= {bound}"
        except ValueError:
            pass
```

**Failing input**: Documentation review (not a runtime failure)

## Reproducing the Bug

```python
import attr

@attr.define
class TestClass:
    value: int = attr.field(validator=attr.validators.gt(5))

TestClass(value=6)

try:
    TestClass(value=5)
    print("BUG: value=5 was accepted (would happen if using operator.ge)")
except ValueError:
    print("Correct: value=5 was rejected (confirms use of operator.gt)")
```

## Why This Is A Bug

The `attr.validators.gt()` function's docstring at line 488 in `attr/validators.py` states:

> "The validator uses `operator.ge` to compare the values."

However, the actual implementation at line 495 is:

```python
return _NumberValidator(val, ">", operator.gt)
```

This uses `operator.gt` (strictly greater than), not `operator.ge` (greater than or equal). The documentation contradicts the implementation, which violates the API contract and could confuse users about the validator's behavior.

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