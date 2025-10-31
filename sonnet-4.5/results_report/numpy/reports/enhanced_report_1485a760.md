# Bug Report: numpy.rec.array IndexError on Empty List Input

**Target**: `numpy.rec.array`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array` crashes with an uninformative `IndexError` when passed an empty list or tuple, instead of handling it gracefully or raising a descriptive error message.

## Property-Based Test

```python
import numpy.rec
from hypothesis import given, strategies as st


@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_array_handles_all_list_sizes(lst):
    result = numpy.rec.array(lst)
    assert isinstance(result, numpy.rec.recarray)

if __name__ == "__main__":
    test_array_handles_all_list_sizes()
```

<details>

<summary>
**Failing input**: `[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 11, in <module>
    test_array_handles_all_list_sizes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 6, in test_array_handles_all_list_sizes
    def test_array_handles_all_list_sizes(lst):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 7, in test_array_handles_all_list_sizes
    result = numpy.rec.array(lst)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 1056, in array
    if isinstance(obj[0], (tuple, list)):
                  ~~~^^^
IndexError: list index out of range
Falsifying example: test_array_handles_all_list_sizes(
    lst=[],
)
```
</details>

## Reproducing the Bug

```python
import numpy.rec

result = numpy.rec.array([])
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/repo.py", line 3, in <module>
    result = numpy.rec.array([])
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 1056, in array
    if isinstance(obj[0], (tuple, list)):
                  ~~~^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Undocumented Restriction**: The function documentation describes it as a "general-purpose record array constructor" that handles a "wide-variety of objects" without any mention that empty lists/tuples are unsupported.

2. **Inconsistent with NumPy Conventions**: Regular `numpy.array([])` handles empty lists gracefully, returning an empty array with shape `(0,)` and dtype `float64`. Users would reasonably expect similar behavior from `numpy.rec.array`.

3. **Uninformative Error**: The crash occurs with a generic `IndexError` rather than a descriptive error message. If empty lists were intentionally unsupported, the code should check and raise a `ValueError` with an explanatory message.

4. **Documentation Implies Non-Empty**: While the documentation states "if the first object is an `~numpy.ndarray`" (emphasis on "the first object"), it doesn't explicitly require non-empty input. This ambiguity combined with the generic error makes it a usability bug.

## Relevant Context

Testing reveals that this is a systemic issue in numpy's record array implementation:
- `numpy.rec.fromrecords([])` also crashes with `IndexError: list index out of range`
- `numpy.rec.fromarrays([])` also crashes with `IndexError: list index out of range`

The crash occurs at line 1056 in `/numpy/_core/records.py` where the code attempts to inspect `obj[0]` without first checking if the list is empty:

```python
elif isinstance(obj, (list, tuple)):
    if isinstance(obj[0], (tuple, list)):  # <-- Crashes here on empty list
        return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
    else:
        return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
```

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.rec.array.html

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1053,6 +1053,9 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

     elif isinstance(obj, (list, tuple)):
+        if len(obj) == 0:
+            # For empty lists/tuples, default to fromrecords which will handle the empty case
+            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         if isinstance(obj[0], (tuple, list)):
             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         else:
```

Note: While this fix prevents the immediate crash in `numpy.rec.array`, the underlying `fromrecords` and `fromarrays` functions will still fail with their own `IndexError`. A complete fix would require updating those functions as well to handle empty input gracefully.