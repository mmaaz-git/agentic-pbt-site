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
```

**Failing input**: `formats=[]`

## Reproducing the Bug

```python
import numpy as np
import numpy.rec

dtype = np.dtype([])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

rec.pprint()
```

Running this produces:
```
ValueError: max() iterable argument is empty
```

## Why This Is A Bug

1. It's valid to create a record with no fields (NumPy allows `dtype=[]`)
2. The `pprint()` method is a public API that should handle all valid record instances
3. The crash is unexpected - users would reasonably expect either an empty string or graceful handling
4. Other record methods work fine with empty records - only `pprint()` crashes

## Fix

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