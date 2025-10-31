# Bug Report: numpy.ctypeslib.ndpointer Duplicate Flags

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ndpointer` incorrectly adds duplicate flags multiple times, causing incorrect validation that rejects valid arrays.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings

@given(
    flag=st.sampled_from(['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
    num_duplicates=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=200)
def test_ndpointer_duplicate_flags_invariant(flag, num_duplicates):
    flags_single = [flag]
    flags_dup = [flag] * num_duplicates

    ptr_single = npc.ndpointer(flags=flags_single)
    ptr_dup = npc.ndpointer(flags=flags_dup)

    assert ptr_single._flags_ == ptr_dup._flags_, \
        f"Duplicate flags should have same effect: {ptr_single._flags_} vs {ptr_dup._flags_}"
```

**Failing input**: `flag='C_CONTIGUOUS', num_duplicates=2`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

ptr_single = npc.ndpointer(flags=['C_CONTIGUOUS'])
ptr_dup = npc.ndpointer(flags=['C_CONTIGUOUS', 'C_CONTIGUOUS'])

print(f"Single flag: _flags_ = {ptr_single._flags_}")
print(f"Duplicate flags: _flags_ = {ptr_dup._flags_}")

arr = np.zeros((2, 3), dtype=np.int32, order='C')
print(f"\nArray flags.num: {arr.flags.num}")

ptr_single.from_param(arr)
print("Single flag: PASS")

ptr_dup.from_param(arr)
```

Output:
```
Single flag: _flags_ = 1
Duplicate flags: _flags_ = 2
Array flags.num: 1285

Single flag: PASS
TypeError: array must have flags ['F_CONTIGUOUS']
```

## Why This Is A Bug

The `_num_fromflags` function adds flag values without checking for duplicates. This causes:
1. `flags=['C_CONTIGUOUS', 'C_CONTIGUOUS']` to have `_flags_ = 2` instead of `1`
2. Validation incorrectly requires `F_CONTIGUOUS` bit to be set
3. Valid C-contiguous arrays are incorrectly rejected

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -134,8 +134,8 @@ def _num_fromflags(flaglist):
     num = 0
     for val in flaglist:
-        num += mu._flagdict[val]
+        num |= mu._flagdict[val]
     return num
```