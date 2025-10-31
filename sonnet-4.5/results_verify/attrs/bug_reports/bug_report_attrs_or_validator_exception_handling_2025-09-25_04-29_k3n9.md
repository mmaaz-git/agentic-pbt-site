# Bug Report: attrs or_ Validator Hides Non-Validation Exceptions

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `or_` validator catches ALL exceptions (including AttributeError, KeyError, etc.) instead of only validation exceptions (ValueError, TypeError). This hides bugs in validator implementations and makes debugging significantly harder.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr
from attr import validators


@given(st.integers())
def test_or_validator_exception_handling(value):
    """
    Property: or_ validator should only catch validation exceptions, not all exceptions.
    """

    class BuggyValidator:
        def __call__(self, inst, attr, value):
            raise AttributeError("Oops! This is a bug, not a validation error")

    buggy = BuggyValidator()
    normal = validators.instance_of(str)
    combined = validators.or_(buggy, normal)

    @attr.define
    class TestClass:
        x: int = attr.field(validator=combined)

    with pytest.raises(AttributeError, match="Oops"):
        TestClass(x=value)
```

**Failing input**: Any integer (e.g., `value=0`)

## Reproducing the Bug

```python
import attr
from attr import validators


class BuggyValidator:
    def __call__(self, inst, attr, value):
        raise AttributeError("Bug in validator implementation!")


combined = validators.or_(BuggyValidator(), validators.instance_of(str))

@attr.define
class TestClass:
    x: int = attr.field(validator=combined)

try:
    TestClass(x=42)
except AttributeError:
    print("GOOD: AttributeError propagated")
except ValueError as e:
    print(f"BUG: or_ hid the AttributeError, raised ValueError instead: {e}")
```

**Output:**
```
BUG: or_ hid the AttributeError, raised ValueError instead: None of (<__main__.BuggyValidator object at 0x...>, <instance_of validator for type <class 'str'>>) satisfied for value 42
```

## Why This Is A Bug

1. **Hides validator bugs**: If a validator has a bug that raises AttributeError, KeyError, etc., the `or_` validator silently swallows it
2. **Inconsistent with `not_` validator**: The `not_` validator (line 632 in validators.py) defaults to only catching `(ValueError, TypeError)`, indicating these are the standard validation exception types
3. **Makes debugging harder**: Developers see a generic "None of ... satisfied" message instead of the actual error
4. **Code smell**: The linter suppressions on line 675 (`# noqa: BLE001, PERF203, S112`) acknowledge this is problematic

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

This aligns with the standard validation exception types used elsewhere in attrs (e.g., `not_` validator) and ensures that unexpected exceptions from buggy validators are properly propagated rather than hidden.