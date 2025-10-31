# Bug Report: attrs.validators.or_ Masks Programming Errors

**Target**: `attrs.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `or_` validator catches ALL exceptions (including NameError, AttributeError, KeyError, etc.), masking programming errors in validators and making debugging extremely difficult. This is inconsistent with the `and_` validator which propagates all exceptions.

## Property-Based Test

```python
import attrs
from attrs import validators
from hypothesis import given, strategies as st


def make_attr():
    return attrs.Attribute(
        name="test", default=None, validator=None, repr=True,
        cmp=None, eq=True, eq_key=None, order=False,
        order_key=None, hash=None, init=True, kw_only=False,
        type=None, converter=None, metadata={}, alias=None
    )


@given(st.integers())
def test_or_should_not_mask_programming_errors(x):
    def buggy_validator(inst, attr, value):
        undefined_variable

    def valid_validator(inst, attr, value):
        pass

    v = validators.or_(buggy_validator, valid_validator)

    v(None, make_attr(), x)
```

**Failing input**: Any integer (e.g., `x=0`)

## Reproducing the Bug

```python
import attrs
from attrs import validators


def buggy_validator(inst, attr, value):
    undefined_variable


def working_validator(inst, attr, value):
    pass


@attrs.define
class Example:
    value: int = attrs.field(
        validator=validators.or_(buggy_validator, working_validator)
    )


obj = Example(value=42)
```

## Why This Is A Bug

1. **Inconsistent with `and_` validator**: The `and_` validator does not catch exceptions - they propagate normally. The `or_` validator should behave similarly.

2. **Makes debugging impossible**: When a validator has a typo or programming error, `or_` silently catches it and continues. This can hide serious bugs for a long time.

3. **Violates principle of least surprise**: Exceptions like NameError, AttributeError, and KeyError indicate programming errors, not validation failures. Users expect these to propagate.

4. **Only validation exceptions should be caught**: The `or_` validator should only catch ValueError and TypeError (the standard validation exceptions), not all Exception types.

## Fix

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -668,11 +668,11 @@ class _OrValidator:
     validators = attrib()

     def __call__(self, inst, attr, value):
         for v in self.validators:
             try:
                 v(inst, attr, value)
-            except Exception:  # noqa: BLE001, PERF203, S112
+            except (ValueError, TypeError):
                 continue
             else:
                 return

         msg = f"None of {self.validators!r} satisfied for value {value!r}"
         raise ValueError(msg)
```