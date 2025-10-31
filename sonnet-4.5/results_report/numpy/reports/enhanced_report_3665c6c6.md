# Bug Report: numpy.rec.array IndexError on Empty List/Tuple Input

**Target**: `numpy.rec.array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array()` crashes with IndexError when passed an empty list or tuple, failing to gracefully handle empty input data which should return an empty record array.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec as rec
import pytest

@given(st.lists(st.lists(st.integers(), min_size=2, max_size=2), min_size=0, max_size=10))
def test_array_handles_empty_input(records):
    records_tuples = [tuple(r) for r in records]
    r = rec.array(records_tuples, formats=['i4', 'i4'], names='x,y')
    assert len(r) == len(records)
```

<details>

<summary>
**Failing input**: `records=[]`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/3
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_array_handles_empty_input FAILED                           [100%]

=================================== FAILURES ===================================
________________________ test_array_handles_empty_input ________________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 6, in test_array_handles_empty_input
  |     def test_array_handles_empty_input(records):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 8, in test_array_handles_empty_input
    |     r = rec.array(records_tuples, formats=['i4', 'i4'], names='x,y')
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 1057, in array
    |     return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 724, in fromrecords
    |     retval = sb.array(recList, dtype=descr)
    | OverflowError: Python integer 2147483648 out of bounds for int32
    | Falsifying example: test_array_handles_empty_input(
    |     records=[[0, 2_147_483_648]],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:33
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 8, in test_array_handles_empty_input
    |     r = rec.array(records_tuples, formats=['i4', 'i4'], names='x,y')
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 1056, in array
    |     if isinstance(obj[0], (tuple, list)):
    |                   ~~~^^^
    | IndexError: list index out of range
    | Falsifying example: test_array_handles_empty_input(
    |     records=[],
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_array_handles_empty_input - ExceptionGroup: Hypothesis f...
============================== 1 failed in 1.01s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy.rec as rec

# Test case 1: Empty list with formats
try:
    print("Test 1: rec.array([], formats=['i4'], names='x')")
    r = rec.array([], formats=['i4'], names='x')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 2: Empty tuple with formats
try:
    print("Test 2: rec.array((), formats=['i4'], names='x')")
    r = rec.array((), formats=['i4'], names='x')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 3: Empty list with multiple fields
try:
    print("Test 3: rec.array([], formats=['i4', 'i4'], names='x,y')")
    r = rec.array([], formats=['i4', 'i4'], names='x,y')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Test 1: rec.array([], formats=['i4'], names='x')
Error: IndexError: list index out of range

Test 2: rec.array((), formats=['i4'], names='x')
Error: IndexError: tuple index out of range

Test 3: rec.array([], formats=['i4', 'i4'], names='x,y')
Error: IndexError: list index out of range
```
</details>

## Why This Is A Bug

The `numpy.rec.array()` function is documented as a "general-purpose record array constructor" that should handle a "wide-variety of objects". Creating record arrays from empty data is a legitimate and common use case that occurs naturally in data processing pipelines:

1. **Filtering operations** - When filtering datasets, the result may legitimately be empty (e.g., no records match the criteria)
2. **Data aggregation** - Collecting results from multiple sources where some sources have no data
3. **API consistency** - Other NumPy array creation functions handle empty inputs gracefully (e.g., `numpy.array([])` works fine)
4. **Documentation expectations** - The function documentation doesn't warn about empty input restrictions

The bug occurs in line 1056 of `numpy/_core/records.py` where the code attempts to access `obj[0]` without first checking if the list/tuple is empty. This violates the principle of defensive programming and creates an inconsistent API where users must add special-case handling for empty data.

## Relevant Context

The bug is located in `/numpy/_core/records.py` at line 1056 in the `array()` function. The problematic code:

```python
elif isinstance(obj, (list, tuple)):
    if isinstance(obj[0], (tuple, list)):  # <-- Line 1056: IndexError if obj is empty
        return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
    else:
        return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
```

The `fromrecords()` and `fromarrays()` functions that get called both handle empty inputs properly when called directly. The issue is only in the dispatch logic of the main `array()` function.

Related NumPy documentation:
- https://numpy.org/doc/stable/reference/generated/numpy.rec.array.html
- https://numpy.org/doc/stable/user/basics.rec.html

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1053,7 +1053,10 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

     elif isinstance(obj, (list, tuple)):
-        if isinstance(obj[0], (tuple, list)):
+        if len(obj) == 0:
+            # Empty input - dispatch to fromrecords which handles empty properly
+            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
+        elif isinstance(obj[0], (tuple, list)):
             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         else:
             return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
```