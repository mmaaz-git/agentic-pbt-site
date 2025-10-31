# Bug Report: numpy.ctypeslib ndpointer Flag Addition Bug

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ndpointer` function incorrectly adds duplicate flags instead of treating them idempotently, causing flag validation to use incorrect values.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st


@given(
    flag=st.sampled_from(['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'OWNDATA']),
    count=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_repeated_flags_are_added_not_ored(flag, count):
    flags_str = ','.join([flag] * count)

    single_ptr = np.ctypeslib.ndpointer(flags=flag)
    multi_ptr = np.ctypeslib.ndpointer(flags=flags_str)

    assert single_ptr._flags_ == multi_ptr._flags_
```

**Failing input**: `flag='C_CONTIGUOUS', count=2`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.int32)

ptr_single = np.ctypeslib.ndpointer(flags="ALIGNED")
ptr_double = np.ctypeslib.ndpointer(flags="ALIGNED,ALIGNED")

assert ptr_single._flags_ == 256
assert ptr_double._flags_ == 512

ptr_single.from_param(arr)

try:
    ptr_double.from_param(arr)
    assert False, "Should reject"
except TypeError:
    print("Bug confirmed: valid aligned array incorrectly rejected")
```

## Why This Is A Bug

Flag values are bit flags that should be combined using bitwise OR, not arithmetic addition. Specifying the same flag multiple times should be idempotent (have the same effect as specifying it once).

The root cause is in `_num_fromflags` function which uses `+=` instead of `|=`:

```python
def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += mu._flagdict[val]  # BUG: should be |=
    return num
```

This can cause `ndpointer` to incorrectly reject valid arrays. For example, if a flag has bit value 1, specifying it twice gives `_flags_ = 2`, which won't match an array that has the flag set (since `array.flags.num & 2 != 2` when only bit 0 is set).

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -168,7 +168,7 @@ def _flags_fromnum(num):
 def _num_fromflags(flaglist):
     num = 0
     for val in flaglist:
-        num += mu._flagdict[val]
+        num |= mu._flagdict[val]
     return num
```