# Bug Report: attr.validators.or_ Silently Catches All Exceptions Including Programming Errors

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.validators.or_()` validator catches ALL exceptions (line 675: `except Exception`) instead of only validation errors, silently suppressing programming errors like NameError, AttributeError, and other bugs in user validators, making debugging impossible.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property-based test for attr.validators.or_ bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr
from hypothesis import given, strategies as st


class ValidatorWithNameError:
    """A validator with a programming error"""
    def __call__(self, inst, attr, value):
        undefined_variable  # This will raise NameError


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
        pass  # This is what we expect


if __name__ == "__main__":
    test_or_validator_should_not_mask_name_errors()
```

<details>

<summary>
**Failing input**: `value=0` (or any integer)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 61, in <module>
    test_or_validator_should_not_mask_name_errors()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 18, in test_or_validator_should_not_mask_name_errors
    def test_or_validator_should_not_mask_name_errors(value):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 55, in test_or_validator_should_not_mask_name_errors
    assert False, "NameError was silently caught"
           ^^^^^
AssertionError: NameError was silently caught
Falsifying example: test_or_validator_should_not_mask_name_errors(
    value=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for attr.validators.or_ bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr


class BuggyValidator:
    """A validator with a programming error (NameError)"""
    def __call__(self, inst, attr, value):
        # This will raise NameError when called
        undefined_variable


# Create an or_ validator that includes our buggy validator
validator = attr.validators.or_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)

# Create a minimal Attribute object for testing
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

# Test with integer value - this SHOULD raise NameError but doesn't
print("Testing or_ validator with buggy validator that has NameError...")
print(f"Input value: 42")
print(f"Expected: NameError: name 'undefined_variable' is not defined")
print(f"Actual: ", end="")

try:
    validator(None, field_attr, 42)
    print("No error raised! The NameError was silently caught and the second validator passed.")
except NameError as e:
    print(f"NameError raised (correct): {e}")
except Exception as e:
    print(f"Different error raised: {type(e).__name__}: {e}")
```

<details>

<summary>
Output showing the bug: NameError is silently suppressed
</summary>
```
Testing or_ validator with buggy validator that has NameError...
Input value: 42
Expected: NameError: name 'undefined_variable' is not defined
Actual: No error raised! The NameError was silently caught and the second validator passed.
```
</details>

## Why This Is A Bug

This violates expected behavior and Python best practices in multiple ways:

1. **Inconsistent with sibling validators in the same library**:
   - `and_()` validator (in `_make.py`) does NOT catch exceptions - it propagates them naturally
   - `not_()` validator explicitly catches only `(ValueError, TypeError)` by default and documents this behavior
   - `or_()` silently catches ALL exceptions without documentation

2. **Violates Python's core philosophy** (PEP 20 - The Zen of Python):
   - "Errors should never pass silently"
   - "Unless explicitly silenced" (which the documentation doesn't mention)
   - "Explicit is better than implicit"

3. **Makes debugging nearly impossible**: When a user writes a validator with a programming error (NameError, AttributeError, ImportError, etc.), the `or_()` validator silently suppresses it and tries the next validator. The user gets no feedback their validator is broken.

4. **Undocumented behavior**: The docstring only mentions raising `ValueError` when no validator is satisfied. There's no mention that ALL exceptions from child validators are caught and suppressed.

5. **Security implications**: Silently catching exceptions can hide critical failures, potentially allowing invalid data through when a validator is broken.

## Relevant Context

The implementation at line 675 in `/attr/validators.py` shows:
```python
def __call__(self, inst, attr, value):
    for v in self.validators:
        try:
            v(inst, attr, value)
        except Exception:  # noqa: BLE001, PERF203, S112
            continue
```

The `noqa` comments disable linting warnings about:
- BLE001: Blind except Exception
- PERF203: Performance issue with broad exception
- S112: Security issue with broad exception

These suppressed warnings indicate the developers knew this was problematic but chose to do it anyway. However, this doesn't make it correct behavior.

For comparison, the `not_()` validator shows the correct pattern:
- It has an `exc_types` parameter (default: `(ValueError, TypeError)`)
- It documents exactly which exceptions are caught
- Other exceptions propagate normally

Documentation: https://www.attrs.org/en/stable/api.html#attr.validators.or_
Source code: https://github.com/python-attrs/attrs/blob/main/src/attr/validators.py

## Proposed Fix

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

This simple change:
- Makes `or_()` consistent with `not_()` validator's exception handling
- Allows programming errors to propagate for debugging
- Still catches validation errors as intended
- Aligns with Python best practices