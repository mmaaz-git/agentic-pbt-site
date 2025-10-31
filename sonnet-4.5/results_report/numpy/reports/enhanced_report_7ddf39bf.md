# Bug Report: numpy.rec.record.pprint crashes on empty records

**Target**: `numpy.rec.record.pprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pprint()` method of `numpy.record` crashes with a ValueError when called on a record with no fields, instead of gracefully handling the empty case.

## Property-Based Test

```python
import numpy as np
import numpy.rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.sampled_from(['i4', 'f8', 'U10']), min_size=0, max_size=5))
@settings(max_examples=100)
def test_pprint_handles_any_number_of_fields(formats):
    if len(formats) == 0:
        dtype = np.dtype([])
    else:
        names = [f'f{i}' for i in range(len(formats))]
        dtype = np.dtype(list(zip(names, formats)))

    arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
    rec = arr[0]

    result = rec.pprint()
    assert isinstance(result, str)

if __name__ == "__main__":
    test_pprint_handles_any_number_of_fields()
```

<details>

<summary>
**Failing input**: `formats=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 22, in <module>
    test_pprint_handles_any_number_of_fields()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 7, in test_pprint_handles_any_number_of_fields
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 18, in test_pprint_handles_any_number_of_fields
    result = rec.pprint()
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 266, in pprint
    maxlen = max(len(name) for name in names)
ValueError: max() iterable argument is empty
Falsifying example: test_pprint_handles_any_number_of_fields(
    formats=[],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/57/hypo.py:10
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.rec

# Create a record with no fields (empty dtype)
dtype = np.dtype([])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

# Try to pretty-print the empty record
print("Attempting to call pprint() on an empty record...")
try:
    result = rec.pprint()
    print(f"Success! Result: '{result}'")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ValueError: max() iterable argument is empty
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 12, in <module>
    result = rec.pprint()
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 266, in pprint
    maxlen = max(len(name) for name in names)
ValueError: max() iterable argument is empty
Attempting to call pprint() on an empty record...
Error: ValueError: max() iterable argument is empty
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Valid input crashes**: NumPy explicitly supports empty dtypes and empty records. Creating `np.dtype([])` is valid, and arrays with this dtype can be created and manipulated normally. The record object itself is valid and functional - only `pprint()` crashes.

2. **Inconsistent with other methods**: Other display methods handle empty records gracefully:
   - `str(rec)` returns `()`
   - `repr(rec)` returns `np.record((), dtype=[])`
   - These methods correctly handle the empty case, while `pprint()` does not.

3. **Undocumented exception**: The `pprint()` docstring states "Pretty-print all fields." but makes no mention that it will raise an exception for records with no fields. The documentation does not indicate that empty records are unsupported.

4. **Violates principle of least surprise**: A formatting/display method should handle all valid inputs without crashing. Users would reasonably expect either an empty string or some representation like "()" for an empty record, not a crash.

5. **Simple oversight in implementation**: The code at line 266 in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py` calls `max()` on a potentially empty sequence without checking if `names` is empty first.

## Relevant Context

The crash occurs in the `pprint()` method implementation at `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py:266`:

```python
def pprint(self):
    """Pretty-print all fields."""
    names = self.dtype.names  # For empty dtype, this is an empty tuple ()
    maxlen = max(len(name) for name in names)  # max() on empty sequence raises ValueError
```

Empty structured arrays/records are valid in NumPy and used in various contexts:
- Dynamic array construction where fields may be added later
- Template or placeholder data structures
- Results from filtering operations that may yield empty sets

Documentation references:
- NumPy structured arrays documentation: https://numpy.org/doc/stable/user/basics.rec.html
- The `numpy.rec` module is part of NumPy's core functionality for working with structured data

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -262,8 +262,11 @@ class record(nt.void):
     def pprint(self):
         """Pretty-print all fields."""
         # pretty-print all fields
         names = self.dtype.names
+        if not names:
+            return ""
         maxlen = max(len(name) for name in names)
         fmt = '%% %ds: %%s' % maxlen
         rows = [fmt % (name, getattr(self, name)) for name in names]
         return "\n".join(rows)
```