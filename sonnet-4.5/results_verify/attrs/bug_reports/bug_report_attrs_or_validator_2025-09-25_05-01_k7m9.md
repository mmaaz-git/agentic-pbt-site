# Bug Report: attrs.validators.or_ Catches All Exceptions Including Programming Errors

**Target**: `attrs.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `or_` validator catches **all** exceptions (line 675: `except Exception:`), including programming errors like `KeyError`, `AttributeError`, etc. This hides bugs in validators and makes debugging extremely difficult. The validator should only catch validation-related exceptions (ValueError, TypeError) like the `not_` validator does.

## Property-Based Test

```python
from attrs.validators import or_, instance_of
from hypothesis import given, strategies as st
import attrs


def buggy_validator(inst, attr, value):
    data = {}
    result = data[value]


@given(st.text())
def test_or_should_not_hide_programming_errors(value):
    validator = or_(buggy_validator, instance_of(int))
    attr = attrs.Attribute(
        name="test", default=None, validator=None, repr=True,
        cmp=None, eq=True, eq_key=None, order=False,
        order_key=None, hash=None, init=True, kw_only=False,
        type=None, converter=None, metadata={}, alias=None
    )

    try:
        validator(None, attr, value)
    except KeyError:
        pass
```

**Failing input**: Any text value (e.g., `"test"`)

## Reproducing the Bug

```python
from attrs.validators import or_, instance_of


def buggy_validator(inst, attr, value):
    data = {}
    return data[value]


validator = or_(buggy_validator, instance_of(int))


class FakeAttr:
    name = "test"


validator(None, FakeAttr(), "missing_key")
```

Expected: KeyError should propagate (it indicates a bug in the validator)
Actual: No exception is raised; KeyError is silently caught and hidden

## Why This Is A Bug

1. **Inconsistency with `not_` validator**: The `not_` validator (lines 631-664) explicitly accepts an `exc_types` parameter that defaults to `(ValueError, TypeError)`. This demonstrates that validators should only catch specific validation exceptions, not all exceptions.

2. **Hides programming errors**: Exceptions like `KeyError`, `AttributeError`, `IndexError`, etc. indicate bugs in validator code, not validation failures. These should propagate to help developers find and fix bugs.

3. **Makes debugging difficult**: When a validator has a bug (e.g., typo in attribute name causing AttributeError), the `or_` validator silently catches it and continues. The developer gets no indication there's a bug.

4. **Violates principle of least surprise**: Validators are expected to either pass (no exception) or fail validation (raise ValueError/TypeError). Catching *all* exceptions violates this contract.

## Fix

Change line 675 in `/attr/validators.py` from catching all exceptions to only catching validation exceptions:

```diff
@attrs(repr=False, slots=True, unsafe_hash=True)
class _OrValidator:
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

This aligns `or_` with the behavior of `not_` and ensures that programming errors in validators are not hidden.