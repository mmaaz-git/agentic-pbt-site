# Bug Report: numpy.rec.format_parser Poor Error Message for Non-String Names

**Target**: `numpy.rec.format_parser`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`format_parser` crashes with an unhelpful `AttributeError` when given non-string field names, instead of raising a clear `TypeError` or `ValueError` explaining the input requirement.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_format_parser_rejects_integer_names(int_names):
    formats = ['i4'] * len(int_names)
    try:
        numpy.rec.format_parser(formats, int_names, [])
        assert False, f"Should have raised error for integer names {int_names}"
    except (TypeError, ValueError):
        pass
    except AttributeError:
        raise AssertionError("Got unhelpful AttributeError instead of clear TypeError/ValueError")
```

**Failing input**: `int_names=[0]`

## Reproducing the Bug

```python
import numpy.rec

parser = numpy.rec.format_parser(['i4', 'i4'], [0, 1], [])
```

This crashes with:
```
AttributeError: 'int' object has no attribute 'strip'
```

## Why This Is A Bug

The docstring for `format_parser` clearly states that `names` should be "str or list/tuple of str". When a user violates this by passing integers, they should receive a clear error message like `TypeError: Field names must be strings, got int at index 0`.

Instead, they get `AttributeError: 'int' object has no attribute 'strip'`, which:
- Doesn't explain what the user did wrong
- Forces users to read the source code to understand the issue
- Is inconsistent with how `numpy.dtype` handles the same error (which raises a clear `TypeError`)

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -154,6 +154,10 @@ class format_parser:

         if names:
             if type(names) in [list, tuple]:
+                # Validate all names are strings
+                for i, name in enumerate(names):
+                    if not isinstance(name, str):
+                        raise TypeError(f"Field names must be strings, got {type(name).__name__} at index {i}")
                 pass
             elif isinstance(names, str):
                 names = names.split(',')
```
