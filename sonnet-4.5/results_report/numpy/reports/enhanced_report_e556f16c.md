# Bug Report: numpy.rec.format_parser Unhelpful AttributeError for Non-String Field Names

**Target**: `numpy.rec.format_parser`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.rec.format_parser` crashes with an unhelpful `AttributeError: 'int' object has no attribute 'strip'` when given non-string field names, instead of raising a clear `TypeError` or `ValueError` that explains the field names must be strings.

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

# Run the test
test_format_parser_rejects_integer_names()
```

<details>

<summary>
**Failing input**: `int_names=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 8, in test_format_parser_rejects_integer_names
    numpy.rec.format_parser(formats, int_names, [])
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 120, in __init__
    self._setfieldnames(names, titles)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 159, in _setfieldnames
    self._names = [n.strip() for n in names[:self._nfields]]
                   ^^^^^^^
AttributeError: 'int' object has no attribute 'strip'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 16, in <module>
    test_format_parser_rejects_integer_names()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 5, in test_format_parser_rejects_integer_names
    def test_format_parser_rejects_integer_names(int_names):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 13, in test_format_parser_rejects_integer_names
    raise AssertionError("Got unhelpful AttributeError instead of clear TypeError/ValueError")
AssertionError: Got unhelpful AttributeError instead of clear TypeError/ValueError
Falsifying example: test_format_parser_rejects_integer_names(
    int_names=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy.rec

# This demonstrates the bug where format_parser crashes with an unhelpful
# AttributeError when given non-string field names, instead of raising
# a clear TypeError or ValueError explaining the input requirement.

try:
    parser = numpy.rec.format_parser(['i4', 'i4'], [0, 1], [])
    print("No error raised - this should not happen!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

    # Show full traceback
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
AttributeError: 'int' object has no attribute 'strip'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 8, in <module>
    parser = numpy.rec.format_parser(['i4', 'i4'], [0, 1], [])
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 120, in __init__
    self._setfieldnames(names, titles)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 159, in _setfieldnames
    self._names = [n.strip() for n in names[:self._nfields]]
                   ^^^^^^^
AttributeError: 'int' object has no attribute 'strip'
Error type: AttributeError
Error message: 'int' object has no attribute 'strip'

Full traceback:
```
</details>

## Why This Is A Bug

The docstring for `numpy.rec.format_parser` explicitly states that the `names` parameter should be "str or list/tuple of str". When a user violates this contract by passing non-string values like integers, they should receive a clear error message explaining what went wrong, such as `TypeError: Field names must be strings, got int at index 0`.

Instead, the function crashes with `AttributeError: 'int' object has no attribute 'strip'` at line 159 in `_setfieldnames` method. This error:
1. Doesn't explain what the user did wrong
2. Exposes internal implementation details (the `.strip()` call)
3. Forces users to read the source code to understand the issue
4. Is inconsistent with how similar numpy functions handle this error

For comparison, `numpy.dtype` handles the same situation correctly, raising `TypeError: First element of field tuple is neither a tuple nor str` when given invalid field names, which immediately tells the user what's wrong.

## Relevant Context

The error occurs in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py` at line 159 in the `_setfieldnames` method:

```python
self._names = [n.strip() for n in names[:self._nfields]]
```

The code assumes all elements in the `names` list are strings and calls `.strip()` on them without validation. The method already has some input validation (lines 152-157) that checks if `names` is a list/tuple or string, but it doesn't validate the contents of the list.

Documentation: The [numpy.rec.format_parser documentation](https://numpy.org/doc/stable/reference/generated/numpy.rec.format_parser.html) clearly specifies that `names` should be "str or list/tuple of str".

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -150,6 +150,10 @@ class format_parser:

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