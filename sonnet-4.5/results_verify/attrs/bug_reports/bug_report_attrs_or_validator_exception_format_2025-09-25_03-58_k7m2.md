# Bug Report: attrs or_() Validator Exception Format

**Target**: `attrs.validators.or_()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `or_()` validator violates the established API contract for validator exceptions by only passing a message string to `ValueError`, while other validators pass structured data (message, attribute, and context).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attrs
from attrs import validators

@given(st.integers())
def test_or_validator_exception_format_consistency(value):
    @attrs.define
    class Example:
        num: int = attrs.field(validator=validators.or_(
            validators.instance_of(str),
            validators.instance_of(list)
        ))

    try:
        Example(num=value)
    except ValueError as e:
        assert len(e.args) >= 2, \
            f"or_() should pass (msg, attr, ...) but only passed {len(e.args)} args"
```

**Failing input**: Any value that fails validation (e.g., `42` when expecting `str` or `list`)

## Reproducing the Bug

```python
import attrs
from attrs import validators

@attrs.define
class Example:
    value: int = attrs.field(validator=validators.in_([1, 2, 3]))

try:
    Example(value=999)
except ValueError as e:
    print("in_() exception args:", len(e.args))
    print("  ", e.args)


@attrs.define
class Example2:
    value: int = attrs.field(validator=validators.or_(
        validators.instance_of(str),
        validators.instance_of(list)
    ))

try:
    Example2(value=999)
except ValueError as e:
    print("or_() exception args:", len(e.args))
    print("  ", e.args)
```

**Output:**
```
in_() exception args: 4
   ('...message...', Attribute(...), (1, 2, 3), 999)

or_() exception args: 1
   ("None of ... satisfied for value 999",)
```

## Why This Is A Bug

**API Contract Violation**: In version 22.1.0, attrs established a standard for validator exceptions. The `in_()` validator documentation explicitly states:

> Raises:
>     ValueError:
>         With a human readable error message, the attribute (of type
>         `attrs.Attribute`), the expected options, and the value it got.

The release notes for v22.1.0 state:

> .. versionchanged:: 22.1.0
>    The ValueError was incomplete until now and only contained the human
>    readable error message. Now it contains all the information that has
>    been promised since 17.1.0.

All validators follow this pattern:
- `instance_of()`: raises `TypeError(msg, attr, type, value)` (line 100-105 in validators.py)
- `in_()`: raises `ValueError(msg, attr, options, value)` (line 246-251)
- `not_()`: raises `ValueError(msg, attr, validator, value, exc_types)` (line 616-625)

However, `or_()` (added in v24.1.0) only raises `ValueError(msg)` (line 680-681 in validators.py):

```python
msg = f"None of {self.validators!r} satisfied for value {value!r}"
raise ValueError(msg)
```

This breaks API consistency and prevents callers from programmatically extracting structured error information.

## Fix

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -677,7 +677,11 @@ class _OrValidator:
             else:
                 return

         msg = f"None of {self.validators!r} satisfied for value {value!r}"
-        raise ValueError(msg)
+        raise ValueError(
+            msg,
+            attr,
+            self.validators,
+            value,
+        )

     def __repr__(self):
```

Additionally, the docstring should be updated to document the exception format:

```diff
@@ -697,8 +701,9 @@ def or_(*validators):

     Raises:
         ValueError:
-            If no validator is satisfied. Raised with a human-readable error
-            message listing all the wrapped validators and the value that
-            failed all of them.
+            If no validator is satisfied. With a human readable error message,
+            the attribute (of type `attrs.Attribute`), the tuple of validators,
+            and the value it got.
```