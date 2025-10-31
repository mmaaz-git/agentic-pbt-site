# Bug Report: attr.validators.or_ Masks Programming Errors

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.validators.or_()` validator catches ALL exceptions (including programming errors like NameError, AttributeError, etc.) instead of only catching validation errors (ValueError, TypeError). This masks bugs in user-written validators and makes debugging extremely difficult.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr
from hypothesis import given, strategies as st


class ValidatorWithNameError:
    """A validator with a programming error"""
    def __call__(self, inst, attr, value):
        undefined_variable


@given(st.integers())
def test_or_validator_should_not_mask_name_errors(value):
    """
    Property: or_ validator should not silently catch programming errors.

    Evidence:
    - and_ validator does not catch exceptions (just calls each validator)
    - not_ validator only catches (ValueError, TypeError) by default
    - or_ validator catches ALL exceptions (line 675 in validators.py)
    """
    validator = attr.validators.or_(
        ValidatorWithNameError(),
        attr.validators.instance_of(int)
    )

    field_attr = attr.Attribute(
        name='value',
        default=attr.NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        eq_key=None,
        order=False,
        order_key=None,
        hash=None,
        init=True,
        metadata={},
        type=None,
        converter=None,
        kw_only=False,
        inherited=False,
        on_setattr=None,
        alias=None
    )

    try:
        validator(None, field_attr, value)
        assert False, "NameError was silently caught"
    except NameError:
        pass
```

**Failing input**: Any integer value (e.g., `42`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr


class BuggyValidator:
    def __call__(self, inst, attr, value):
        undefined_variable


validator = attr.validators.or_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)

field_attr = attr.Attribute(
    name='test',
    default=attr.NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    eq=True,
    eq_key=None,
    order=False,
    order_key=None,
    hash=None,
    init=True,
    metadata={},
    type=None,
    converter=None,
    kw_only=False,
    inherited=False,
    on_setattr=None,
    alias=None
)

validator(None, field_attr, 42)
```

**Expected**: `NameError: name 'undefined_variable' is not defined`
**Actual**: No error (the second validator passes, masking the NameError)

## Why This Is A Bug

1. **Inconsistent with other validators**: The `and_` validator (in `_make.py`) does NOT catch exceptions - it lets them propagate naturally. The `not_` validator specifically only catches `(ValueError, TypeError)` by default, showing the library's intent to distinguish validation errors from programming errors.

2. **Violates Python principles**: Python's philosophy is "Errors should never pass silently." Catching all exceptions masks real bugs.

3. **Makes debugging impossible**: If a user writes a buggy validator with a NameError, AttributeError, or other programming error, the `or_` validator will silently catch it and try the next validator. The user gets no feedback that their validator is broken.

4. **Undocumented behavior**: The docstring for `or_()` only mentions that it raises `ValueError`, but doesn't document that it catches ALL exception types.

## Fix

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -671,7 +671,7 @@ class _OrValidator:
     def __call__(self, inst, attr, value):
         for v in self.validators:
             try:
                 v(inst, attr, value)
-            except Exception:  # noqa: BLE001, PERF203, S112
+            except (ValueError, TypeError):
                 continue
             else:
                 return
```

This change makes `or_` consistent with `not_` validator's exception handling and prevents masking programming errors while still allowing validators to reject values normally.