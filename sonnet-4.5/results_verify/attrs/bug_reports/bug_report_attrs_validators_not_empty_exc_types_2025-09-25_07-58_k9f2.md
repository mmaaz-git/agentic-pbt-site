# Bug Report: attrs validators.not_ Incomplete Inversion with Empty exc_types

**Target**: `attr.validators.not_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validators.not_()` function fails to properly invert validators when passed an empty `exc_types` tuple, causing exceptions from the wrapped validator to propagate instead of being suppressed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr
from attr import validators
import pytest

@given(st.text())
def test_not_validator_with_empty_exc_types_inverts_completely(value):
    v = validators.not_(
        validators.instance_of(int),
        exc_types=()
    )

    attr_obj = attr.Attribute(
        name="test", default=None, validator=None, repr=True,
        cmp=None, eq=True, eq_key=None, order=False,
        order_key=None, hash=None, init=True, kw_only=False,
        type=None, converter=None, metadata={}, alias=None
    )

    if isinstance(value, int):
        with pytest.raises(ValueError):
            v(None, attr_obj, value)
    else:
        v(None, attr_obj, value)
```

**Failing input**: Any non-integer value like `"string"` causes TypeError to propagate instead of being suppressed.

## Reproducing the Bug

```python
import attr
from attr import validators

@attr.s
class TestClass:
    value = attr.ib(validator=validators.not_(
        validators.instance_of(int),
        exc_types=()
    ))

try:
    obj = TestClass(value="hello")
    print("Created object - unexpected!")
except ValueError as e:
    print(f"ValueError raised: {e}")
except TypeError as e:
    print(f"TypeError propagated (BUG): {e}")
    print("Expected: TypeError to be suppressed, validator to pass")
```

Output:
```
TypeError propagated (BUG): 'value' must be <class 'int'> (got 'hello' that is a <class 'str'>).
Expected: TypeError to be suppressed, validator to pass
```

## Why This Is A Bug

According to the `not_` validator documentation, it "wraps and logically 'inverts' the validator." However, when `exc_types=()` is passed:

1. **Values that pass the wrapped validator** → ValueError is raised ✓ (correctly inverted)
2. **Values that fail the wrapped validator** → Exception propagates ✗ (NOT inverted)

The issue is in `attr/validators.py` lines 610-625:

```python
def __call__(self, inst, attr, value):
    try:
        self.validator(inst, attr, value)
    except self.exc_types:  # When exc_types=(), this is except ():
        pass  # suppress error to invert validity
    else:
        raise ValueError(...)
```

When `self.exc_types = ()`, the `except ():` clause catches **no exceptions**. Therefore, any exception raised by the wrapped validator propagates instead of being suppressed, breaking the inversion property.

This makes the validator's behavior inconsistent and violates the principle of least surprise. Users who pass `exc_types=()` expecting no filtering of exception types end up with a validator that only partially inverts.

## Fix

Option 1: Validate that `exc_types` is non-empty:

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -660,6 +660,10 @@ def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
     try:
         exc_types = tuple(exc_types)
     except TypeError:
         exc_types = (exc_types,)
+
+    if not exc_types:
+        msg = "exc_types must not be empty"
+        raise ValueError(msg)
+
     return _NotValidator(validator, msg, exc_types)
```

Option 2: Use `Exception` as a catch-all when empty:

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -660,6 +660,9 @@ def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
     try:
         exc_types = tuple(exc_types)
     except TypeError:
         exc_types = (exc_types,)
+
+    if not exc_types:
+        exc_types = (Exception,)
+
     return _NotValidator(validator, msg, exc_types)
```