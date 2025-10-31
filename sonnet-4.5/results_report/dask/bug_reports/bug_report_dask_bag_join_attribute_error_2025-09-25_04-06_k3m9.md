# Bug Report: dask.bag.Bag.join AttributeError in Error Message

**Target**: `dask.bag.Bag.join`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `join` method in dask.bag has a bug in its error handling code that causes an `AttributeError` when trying to raise a `TypeError` for invalid input. The error message tries to access `type(other).__name` instead of `type(other).__name__`, which is not a valid attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db


@given(st.integers())
def test_join_error_message_with_invalid_type(invalid_input):
    """
    Property: Error messages should be displayable without crashing.

    This test verifies that when join() is called with an invalid type,
    it should raise a TypeError with a proper error message, not crash
    with an AttributeError.
    """
    bag = db.from_sequence([1, 2, 3])

    try:
        bag.join(invalid_input, lambda x: x)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Joined argument must be" in str(e)
    except AttributeError as e:
        if "'type' object has no attribute '__name'" in str(e):
            raise AssertionError(
                f"Bug found! Error message construction failed with AttributeError: {e}"
            )
```

**Failing input**: Any integer value (e.g., `42`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db

bag = db.from_sequence([1, 2, 3])

try:
    result = bag.join(42, lambda x: x)
except AttributeError as e:
    print(f"AttributeError caught: {e}")
    print(f"Expected TypeError but got AttributeError instead!")
except TypeError as e:
    print(f"TypeError (expected): {e}")
```

**Expected output**: `TypeError` with message about invalid argument type

**Actual output**: `AttributeError: 'type' object has no attribute '__name'`

## Why This Is A Bug

This bug occurs in the error handling path of the `join` method when validating input types. The issue is in line 1203 of `dask/bag/core.py`:

```python
msg = (
    "Joined argument must be single-partition Bag, "
    " delayed object, or Iterable, got %s" % type(other).__name
)
raise TypeError(msg)
```

The problem:
1. **Incorrect attribute access**: `type(other).__name` should be `type(other).__name__`
2. **Double underscore convention**: Python's built-in type attributes use double underscores on both sides (e.g., `__name__`, `__class__`, etc.)
3. **AttributeError instead of TypeError**: When this code path is reached, instead of getting a helpful TypeError, users get a confusing AttributeError about the missing `__name` attribute
4. **Obscures the real error**: The actual issue (invalid argument type) gets masked by this implementation bug

This violates the principle that error messages should be clear and helpful, not cause additional errors.

## Impact Assessment

**Severity: Medium**
- **User Impact**: Users calling `join()` with invalid input types get a confusing `AttributeError` instead of a clear `TypeError`
- **Frequency**: Only affects users who pass invalid types to `join()`, which should be caught during development/testing
- **Debuggability**: Makes debugging harder because the error message doesn't clearly indicate what went wrong
- **Documentation**: The error message is meant to guide users to correct usage, but it fails to do so

## Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -1200,7 +1200,7 @@ class Bag(DaskMethodsMixin):
         elif not isinstance(other, Iterable):
             msg = (
                 "Joined argument must be single-partition Bag, "
-                " delayed object, or Iterable, got %s" % type(other).__name
+                " delayed object, or Iterable, got %s" % type(other).__name__
             )
             raise TypeError(msg)
```