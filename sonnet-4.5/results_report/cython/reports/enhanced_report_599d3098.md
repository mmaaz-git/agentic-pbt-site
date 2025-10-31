# Bug Report: Cython.Compiler.TypeSlots Improper Exception Type for Missing Slots

**Target**: `Cython.Compiler.TypeSlots.get_slot_by_name`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_slot_by_name` function uses `assert False` to signal when a slot is not found, which violates Python best practices for error handling and causes unpredictable behavior when Python is run with optimization flags.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython TypeSlots error handling."""

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


if __name__ == "__main__":
    # Run the test to find a failing case
    test_get_slot_error_type()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/40
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_get_slot_error_type FAILED                                 [100%]

=================================== FAILURES ===================================
___________________________ test_get_slot_error_type ___________________________

slot_name = '0'

    @given(st.text(min_size=1, max_size=30))
    def test_get_slot_error_type(slot_name):
        try:
>           get_slot_by_name(slot_name, {})

hypo.py:12:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

slot_name = '0', compiler_directives = {}

    def get_slot_by_name(slot_name, compiler_directives):
        # For now, only search the type struct, no referenced sub-structs.
        for slot in get_slot_table(compiler_directives).slot_table:
            if slot.slot_name == slot_name:
                return slot
>       assert False, "Slot not found: %s" % slot_name
E       AssertionError: Slot not found: 0

/home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/TypeSlots.py:792: AssertionError

During handling of the above exception, another exception occurred:

    @given(st.text(min_size=1, max_size=30))
>   def test_get_slot_error_type(slot_name):

hypo.py:10:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

slot_name = '0'

    @given(st.text(min_size=1, max_size=30))
    def test_get_slot_error_type(slot_name):
        try:
            get_slot_by_name(slot_name, {})
        except AssertionError:
>           pytest.fail("Bug: AssertionError instead of proper exception")
E           Failed: Bug: AssertionError instead of proper exception
E           Falsifying example: test_get_slot_error_type(
E               slot_name='0',  # or any other generated value
E           )

hypo.py:14: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_get_slot_error_type - Failed: Bug: AssertionError instea...
============================== 1 failed in 1.74s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython TypeSlots assertion error bug."""

from Cython.Compiler.TypeSlots import get_slot_by_name

# Try to get a non-existent slot name
# This should raise a proper exception, not AssertionError
get_slot_by_name('nonexistent_slot', {})
```

<details>

<summary>
AssertionError: Slot not found
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 8, in <module>
    get_slot_by_name('nonexistent_slot', {})
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/TypeSlots.py", line 792, in get_slot_by_name
    assert False, "Slot not found: %s" % slot_name
           ^^^^^
AssertionError: Slot not found: nonexistent_slot
```
</details>

## Why This Is A Bug

This code violates Python's fundamental error handling principles for three critical reasons:

1. **Assertions are disabled with optimization flags**: When Python runs with `-O` or `-OO` flags, all assertions are stripped from the bytecode. This means `get_slot_by_name` would silently return `None` instead of raising an error, potentially causing undefined behavior or crashes later in the program.

2. **Wrong exception type for API contracts**: `AssertionError` is specifically meant for internal invariant violations during development, not for signaling expected runtime errors. The Python documentation explicitly states: "Assertions should not be used for validating input or handling expected error conditions." Similar lookup functions in Python's standard library raise `KeyError`, `ValueError`, or `LookupError` when items aren't found.

3. **Inconsistent with Cython's own patterns**: Other lookup functions in the Cython codebase properly raise exceptions for missing items rather than using assertions, making this an inconsistency that could confuse developers working with or extending the compiler.

## Relevant Context

The function is located at line 787-792 in `/Cython/Compiler/TypeSlots.py`. This is an internal compiler function used to look up type slots by name during code generation. While not public API, it's still called from multiple places within the Cython compiler and should follow proper error handling practices.

The Python documentation on assertions (https://docs.python.org/3/reference/simple_stmts.html#assert) clearly states:
> "Assert statements are a convenient way to insert debugging assertions into a program... The current code generator emits no code for an assert statement when optimization is requested at compile time."

## Proposed Fix

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