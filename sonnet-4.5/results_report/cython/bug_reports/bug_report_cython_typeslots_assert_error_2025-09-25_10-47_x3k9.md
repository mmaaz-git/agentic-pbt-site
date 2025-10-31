# Bug Report: Cython.Compiler.TypeSlots Assertion Error Handling

**Target**: `Cython.Compiler.TypeSlots.get_slot_by_name`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_slot_by_name` function uses `assert False` to signal an error condition (slot not found). This is incorrect because assertions can be disabled with `python -O` and should not be used for error handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.TypeSlots import get_slot_by_name
import pytest


@given(st.text(min_size=1, max_size=30))
def test_get_slot_error_type(slot_name):
    try:
        get_slot_by_name(slot_name, {})
    except AssertionError:
        pytest.fail("Bug: AssertionError instead of proper exception")
    except (ValueError, KeyError, LookupError):
        pass
```

**Failing input**: Any nonexistent slot name, e.g., `'nonexistent_slot'`

## Reproducing the Bug

```python
from Cython.Compiler.TypeSlots import get_slot_by_name

get_slot_by_name('nonexistent_slot', {})
```

Output: `AssertionError: Slot not found: nonexistent_slot`

## Why This Is A Bug

Line 792 in TypeSlots.py:
```python
assert False, "Slot not found: %s" % slot_name
```

Problems:
1. **Disabled with `-O`**: Running `python -O` disables assertions, causing the function to return `None` instead of raising an error
2. **Wrong exception type**: `AssertionError` is meant for invariant violations, not error handling
3. **Inconsistent API**: Other lookup functions typically raise `KeyError`, `ValueError`, or `LookupError` for missing items

This violates Python best practices: assertions are for debugging invariants, not for validating input or signaling errors to callers.

## Fix

```diff
--- a/Cython/Compiler/TypeSlots.py
+++ b/Cython/Compiler/TypeSlots.py
@@ -789,7 +789,7 @@ def get_slot_by_name(slot_name, compiler_directives):
     for slot in get_slot_table(compiler_directives).slot_table:
         if slot.slot_name == slot_name:
             return slot
-    assert False, "Slot not found: %s" % slot_name
+    raise LookupError(f"Slot not found: {slot_name}")
```