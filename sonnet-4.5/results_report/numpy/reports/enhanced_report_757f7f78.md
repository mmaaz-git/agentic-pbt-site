# Bug Report: numpy.rec.fromrecords IndexError with Empty Tuples

**Target**: `numpy.rec.fromrecords`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords` crashes with an IndexError when given a list of empty tuples during automatic dtype inference, despite NumPy supporting empty structured arrays when dtype is explicitly specified.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec


@given(st.lists(st.tuples(), min_size=1, max_size=10))
def test_fromrecords_empty_tuples(records):
    rec_arr = numpy.rec.fromrecords(records)
    assert len(rec_arr) == len(records)


if __name__ == "__main__":
    test_fromrecords_empty_tuples()
```

<details>

<summary>
**Failing input**: `[()]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in <module>
    test_fromrecords_empty_tuples()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 6, in test_fromrecords_empty_tuples
    def test_fromrecords_empty_tuples(records):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 7, in test_fromrecords_empty_tuples
    rec_arr = numpy.rec.fromrecords(records)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 713, in fromrecords
    return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                      titles=titles, aligned=aligned, byteorder=byteorder)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 624, in fromarrays
    shape = arrayList[0].shape
            ~~~~~~~~~^^^
IndexError: list index out of range
Falsifying example: test_fromrecords_empty_tuples(
    records=[()],
)
```
</details>

## Reproducing the Bug

```python
import numpy.rec

records = [(), (), ()]
rec_arr = numpy.rec.fromrecords(records)
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/repo.py", line 4, in <module>
    rec_arr = numpy.rec.fromrecords(records)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 713, in fromrecords
    return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                      titles=titles, aligned=aligned, byteorder=byteorder)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 624, in fromarrays
    shape = arrayList[0].shape
            ~~~~~~~~~^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This is a bug because NumPy explicitly supports empty structured arrays throughout the library. The crash occurs due to an implementation oversight in the automatic dtype inference path, not because empty records are unsupported:

1. **Empty structured arrays are valid in NumPy**: `np.zeros(5, dtype=[])` successfully creates an array with no fields
2. **The function works with explicit dtype**: `np.rec.fromrecords([(), (), ()], dtype=[])` succeeds perfectly
3. **Implementation error, not validation error**: The crash is an `IndexError` from accessing `arrayList[0]` without checking if the list is empty, rather than a `ValueError` indicating invalid input
4. **Inconsistent with NumPy's design**: `np.array([(), (), ()], dtype=[])` works correctly, showing that empty records are intended to be supported

## Relevant Context

The bug occurs in the automatic dtype inference path at line 624 of `/numpy/_core/records.py`. When `fromrecords` receives empty tuples without an explicit dtype, it creates an empty `arrayList` at line 711 (`range(obj.shape[-1])` returns an empty range when tuples are empty). This empty list is then passed to `fromarrays`, which unconditionally accesses `arrayList[0].shape` without checking if the list is non-empty.

Interestingly, `numpy.rec.fromarrays([])` also fails with the same error, contrary to what the initial bug report claimed. This indicates the root cause is in `fromarrays` itself, not just in how `fromrecords` calls it.

Documentation references:
- NumPy structured arrays documentation confirms empty dtypes are valid
- The `fromrecords` documentation doesn't explicitly prohibit empty tuples
- Empty structured types have valid use cases (e.g., placeholder arrays, data with metadata but no fields)

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,10 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     shape = _deprecate_shape_0_as_None(shape)

     if shape is None:
-        shape = arrayList[0].shape
+        if len(arrayList) == 0:
+            shape = (0,)
+        else:
+            shape = arrayList[0].shape
     elif isinstance(shape, int):
         shape = (shape,)
```