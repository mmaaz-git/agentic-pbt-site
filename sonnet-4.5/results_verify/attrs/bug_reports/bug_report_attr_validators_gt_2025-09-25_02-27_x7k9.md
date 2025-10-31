# Bug Report: attr.validators.gt Documentation Error

**Target**: `attr.validators.gt`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attr.validators.gt` function's docstring incorrectly states it uses `operator.ge` when it actually uses `operator.gt`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

@given(st.integers())
def test_gt_validator_uses_correct_operator(bound):
    validator = attr.validators.gt(bound)

    @attr.define
    class TestClass:
        value: int = attr.field(validator=validator)

    try:
        TestClass(bound)
        passed_at_bound = True
    except ValueError:
        passed_at_bound = False

    try:
        TestClass(bound + 1)
        passed_above_bound = True
    except ValueError:
        passed_above_bound = False

    assert not passed_at_bound, f"gt({bound}) should reject value == {bound}"
    assert passed_above_bound, f"gt({bound}) should accept value == {bound + 1}"
```

**Failing input**: Any value, since the implementation is correct but documentation is wrong.

## Reproducing the Bug

```python
import attr
import inspect

print("gt validator docstring:")
print(inspect.getsource(attr.validators.gt))

validator = attr.validators.gt(10)
print(f"\nActual compare_func: {validator.compare_func}")

import operator
print(f"Is operator.gt? {validator.compare_func is operator.gt}")
print(f"Is operator.ge? {validator.compare_func is operator.ge}")

@attr.define
class TestClass:
    value: int = attr.field(validator=attr.validators.gt(10))

try:
    TestClass(10)
    print("\n10 passed validation (BUG if this happens)")
except ValueError:
    print("\n10 failed validation (correct behavior for >)")

try:
    TestClass(11)
    print("11 passed validation (correct behavior for >)")
except ValueError:
    print("11 failed validation (BUG if this happens)")
```

## Why This Is A Bug

The docstring at line 488 of `/attr/validators.py` states:

> "The validator uses `operator.ge` to compare the values."

However, line 495 shows the actual implementation:

```python
return _NumberValidator(val, ">", operator.gt)
```

The validator correctly uses `operator.gt` (strict greater-than), not `operator.ge` (greater-than-or-equal). This documentation error could mislead developers about the validator's behavior, causing them to expect that values equal to the bound would be accepted when they are actually rejected.

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