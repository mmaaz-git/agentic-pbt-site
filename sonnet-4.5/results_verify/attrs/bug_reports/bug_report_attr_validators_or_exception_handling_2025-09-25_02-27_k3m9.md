# Bug Report: attr.validators.or_ Catches All Exceptions Including Programming Errors

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.validators.or_` validator catches ALL exceptions (including programming errors like `NameError`, `AttributeError`, etc.) when testing validators, masking bugs in validator code and making debugging difficult.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

class BuggyValidator:
    def __call__(self, inst, attr, value):
        undefined_variable

@given(st.integers())
def test_or_validator_should_not_mask_programming_errors(value):
    validator = attr.validators.or_(
        BuggyValidator(),
        attr.validators.instance_of(int)
    )

    @attr.define
    class TestClass:
        x: int = attr.field(validator=validator)

    TestClass(value)
```

**Failing input**: Any integer (e.g., `42`)

## Reproducing the Bug

```python
import attr

class BuggyValidator:
    def __call__(self, inst, attr, value):
        undefined_variable

@attr.define
class TestClass:
    value: int = attr.field(
        validator=attr.validators.or_(
            BuggyValidator(),
            attr.validators.instance_of(int)
        )
    )

obj = TestClass(42)
print(f"Created object with value: {obj.value}")
print("The NameError from BuggyValidator was silently swallowed!")
```

Expected behavior: `NameError: name 'undefined_variable' is not defined`

Actual behavior: Object is created successfully, masking the programming error.

## Why This Is A Bug

In `/attr/validators.py` lines 671-678, the `_OrValidator` catches all exceptions:

```python
def __call__(self, inst, attr, value):
    for v in self.validators:
        try:
            v(inst, attr, value)
        except Exception:  # Catches ALL exceptions!
            continue
        else:
            return
```

This masks legitimate programming errors such as:
- `NameError` (undefined variables)
- `AttributeError` (missing attributes)
- `KeyError` (missing keys)
- `ImportError` (missing modules)
- Any other bugs in validator code

The validator should only catch validation-related exceptions (`ValueError`, `TypeError`), consistent with the `not_` validator which defaults to catching only `(ValueError, TypeError)` (line 631).

Masking programming errors makes debugging extremely difficult because:
1. Real bugs are hidden behind generic "validation failed" messages
2. Developers won't know their validator code has bugs
3. The actual error location and type are lost

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

This change ensures that only validation exceptions are caught, while programming errors properly propagate for debugging.