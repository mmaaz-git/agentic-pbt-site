# Bug Report: attrs or_ Validator Incorrectly Catches All Exceptions Instead of Just Validation Exceptions

**Target**: `attr.validators.or_`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `or_` validator in attrs catches ALL exceptions (including AttributeError, KeyError, etc.) instead of only validation exceptions (ValueError, TypeError), which hides programming errors in validator implementations and makes debugging significantly harder.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test demonstrating the or_ validator exception handling bug.
Property: or_ validator should only catch validation exceptions, not all exceptions.
"""

from hypothesis import given, strategies as st, settings
import attr
from attr import validators
import pytest


@given(st.integers())
@settings(max_examples=1)  # We only need one example to demonstrate the bug
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


if __name__ == "__main__":
    # Run the test to demonstrate the failure
    try:
        test_or_validator_exception_handling()
        print("Test passed - no bug detected")
    except AssertionError as e:
        print(f"Test failed - bug confirmed!")
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"Test execution error: {type(e).__name__}: {e}")
```

<details>

<summary>
**Failing input**: `value=0`
</summary>
```
Test execution error: ValueError: None of (<__main__.test_or_validator_exception_handling.<locals>.BuggyValidator object at 0x79edaa185be0>, <instance_of validator for type <class 'str'>>) satisfied for value 0
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for attrs or_ validator exception handling bug.
This demonstrates that or_ catches ALL exceptions instead of just validation exceptions.
"""

import attr
from attr import validators


class BuggyValidator:
    """A validator with a programming error that raises AttributeError."""
    def __call__(self, inst, attr, value):
        # This simulates a programming error in the validator
        raise AttributeError("Bug in validator implementation!")


# Create an or_ validator combining the buggy validator with a normal one
combined = validators.or_(BuggyValidator(), validators.instance_of(str))

@attr.define
class TestClass:
    x: int = attr.field(validator=combined)

# Try to create an instance - the AttributeError should propagate but doesn't
print("Testing or_ validator exception handling with value=42")
print("-" * 60)

try:
    TestClass(x=42)
    print("ERROR: No exception was raised!")
except AttributeError as e:
    print(f"GOOD: AttributeError propagated correctly")
    print(f"  Message: {e}")
except ValueError as e:
    print(f"BUG: or_ hid the AttributeError, raised ValueError instead")
    print(f"  Message: {e}")
except Exception as e:
    print(f"UNEXPECTED: Got {type(e).__name__}: {e}")
```

<details>

<summary>
BUG: or_ hid the AttributeError, raised ValueError instead
</summary>
```
Testing or_ validator exception handling with value=42
------------------------------------------------------------
BUG: or_ hid the AttributeError, raised ValueError instead
  Message: None of (<__main__.BuggyValidator object at 0x77e5ef21c830>, <instance_of validator for type <class 'str'>>) satisfied for value 42
```
</details>

## Why This Is A Bug

This behavior violates expected exception handling patterns in attrs and Python best practices:

1. **Inconsistent with Library Patterns**: Every other validator in attrs only raises or catches specific validation exceptions (ValueError and TypeError). For example:
   - `instance_of()` raises TypeError for type mismatches (line 100 in validators.py)
   - `in_()` raises ValueError for values not in options (line 139)
   - All comparison validators (`lt`, `le`, `gt`, `ge`) raise ValueError
   - Length validators (`max_len`, `min_len`) raise ValueError

2. **Explicit Contract in `not_` Validator**: The `not_` validator (line 631) explicitly defaults to `exc_types=(ValueError, TypeError)`, establishing that these are the standard validation exception types. Its documentation states: "Exception type(s) to capture. Other types raised by child validators will not be intercepted and pass through."

3. **Hides Programming Errors**: When a validator has a bug (e.g., AttributeError from accessing a missing attribute, KeyError from dictionary access, NameError from undefined variables), these errors are silently swallowed and converted to a generic "None of ... satisfied" message.

4. **Code Acknowledges the Problem**: The problematic line (675) includes three linter suppressions: `# noqa: BLE001, PERF203, S112`
   - BLE001: Blind except (catching Exception)
   - S112: Try-except-continue
   - PERF203: Performance warning

   These suppressions indicate awareness that catching all exceptions is problematic.

5. **Debugging Impact**: Developers cannot identify the actual error in their custom validators, making it extremely difficult to debug validator implementations.

## Relevant Context

The `or_` validator implementation is located at `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/validators.py` lines 671-681.

The current implementation:
```python
def __call__(self, inst, attr, value):
    for v in self.validators:
        try:
            v(inst, attr, value)
        except Exception:  # noqa: BLE001, PERF203, S112
            continue
        else:
            return

    msg = f"None of {self.validators!r} satisfied for value {value!r}"
    raise ValueError(msg)
```

This pattern of catching all exceptions is considered an anti-pattern in Python because it prevents proper error propagation and debugging. The Python documentation and style guides consistently recommend catching specific exceptions.

Documentation reference: The attrs documentation doesn't explicitly state that `or_` should catch all exceptions, and the established pattern from other validators indicates this is unintended behavior.

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

This fix aligns the `or_` validator with the established validation exception pattern used throughout attrs, ensuring that programming errors are properly propagated while still catching legitimate validation failures.