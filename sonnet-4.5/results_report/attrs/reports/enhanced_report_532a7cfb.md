# Bug Report: attr.validators.or_ Silently Swallows Programming Errors During Validation

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.validators.or_` validator catches ALL exceptions including programming errors (NameError, AttributeError, etc.), making debugging impossible by silently swallowing bugs in validator code.

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

# Run the test
test_or_validator_should_not_mask_programming_errors()
```

<details>

<summary>
**Failing input**: `42`
</summary>
```
Test passed without raising NameError.
The test ran successfully with value=42 and many other random integers.
Expected: NameError: name 'undefined_variable' is not defined
Actual: Test passed, NameError was silently swallowed by or_() validator.
```
</details>

## Reproducing the Bug

```python
import attr

class BuggyValidator:
    def __call__(self, inst, attr, value):
        # This should raise a NameError
        undefined_variable

@attr.define
class TestClass:
    value: int = attr.field(
        validator=attr.validators.or_(
            BuggyValidator(),
            attr.validators.instance_of(int)
        )
    )

# This should crash with NameError but doesn't
obj = TestClass(42)
print(f"Created object with value: {obj.value}")
print("The NameError from BuggyValidator was silently swallowed!")
```

<details>

<summary>
Successfully creates object despite programming error
</summary>
```
Created object with value: 42
The NameError from BuggyValidator was silently swallowed!
```
</details>

## Why This Is A Bug

The `or_()` validator implementation in `/attr/validators.py` (lines 671-678) catches ALL exceptions indiscriminately:

```python
def __call__(self, inst, attr, value):
    for v in self.validators:
        try:
            v(inst, attr, value)
        except Exception:  # Catches ALL exceptions including programming errors
            continue
        else:
            return
```

This violates expected behavior because:

1. **Inconsistency with attrs' own patterns**: The `not_()` validator in the same module (line 631) explicitly limits exception catching to `(ValueError, TypeError)` by default, establishing a pattern that validation errors should be distinguished from programming errors.

2. **Breaks Python debugging conventions**: Python's standard practice is to let programming errors (NameError, AttributeError, KeyError, ImportError) propagate for debugging. The code even has suppressed linting warnings (`# noqa: BLE001, PERF203, S112`) acknowledging this is against best practices.

3. **Documentation doesn't specify this behavior**: The docstring only mentions raising `ValueError` when no validator is satisfied. It doesn't document that ALL exceptions will be caught, leaving users unaware of this dangerous behavior.

4. **Makes debugging impossible**: When a custom validator has a programming error (undefined variable, typo, missing import), the error is silently swallowed. Developers get no feedback that their code is broken, making it extremely difficult to debug validator logic.

## Relevant Context

The `not_()` validator in the same file demonstrates the correct approach (line 631):
```python
def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
```

It has an `exc_types` parameter with documentation stating: "Exception type(s) to capture. Other types raised by child validators will not be intercepted and pass through."

This shows that attrs already has an established pattern for selective exception catching that distinguishes between validation failures and programming errors. The `or_()` validator's behavior is inconsistent with this pattern.

Documentation reference: https://www.attrs.org/en/stable/api.html#attr.validators.or_

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

This change aligns `or_()` with the established pattern in `not_()`, catching only validation-related exceptions while allowing programming errors to propagate for proper debugging.