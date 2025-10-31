# Bug Report: numpy.char partition/rpartition Lose Null Characters

**Target**: `numpy.char.partition`, `numpy.char.rpartition`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `partition` and `rpartition` functions lose null characters (`'\x00'`) from strings, violating the fundamental property that partitioning should preserve the original string content when parts are concatenated.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.char as nc

@given(st.text(min_size=1), st.text(min_size=1))
def test_partition_preserves_content(s, sep):
    arr = np.array([s])
    partitioned = nc.partition(arr, sep)
    parts = partitioned[0]
    reconstructed = parts[0] + parts[1] + parts[2]
    assert reconstructed == s
```

**Failing input**: `s='\x00'`, `sep='0'` (or any separator)

## Reproducing the Bug

```python
import numpy as np
import numpy.char as nc

s = '\x00'
arr = np.array([s])

partitioned = nc.partition(arr, '0')
parts = (partitioned[0][0], partitioned[0][1], partitioned[0][2])
reconstructed = ''.join(parts)

print(f"Original:      {s!r}")
print(f"Partition:     {parts}")
print(f"Reconstructed: {reconstructed!r}")
print(f"Content preserved: {reconstructed == s}")

print(f"\nPython's partition: {s.partition('0')}")

rpartitioned = nc.rpartition(arr, 'x')
r_parts = (rpartitioned[0][0], rpartitioned[0][1], rpartitioned[0][2])
print(f"\nrpartition result: {r_parts}")
print(f"Reconstructed: {''.join(r_parts)!r}")
```

Output:
```
Original:      '\x00'
Partition:     ('', '', '')
Reconstructed: ''
Content preserved: False

Python's partition: ('\x00', '', '')

rpartition result: ('', '', '')
Reconstructed: ''
```

## Why This Is A Bug

The documentation states that partition "return 3 strings containing the part before the separator, the separator itself, and the part after the separator. If the separator is not found, return 3 strings containing the string itself, followed by two empty strings."

When the separator is not found in `'\x00'`, it should return `('\x00', '', '')` like Python does, but numpy.char returns `('', '', '')`, losing the null character entirely.

This violates the fundamental contract that `parts[0] + parts[1] + parts[2]` should equal the original string.

## Fix

The issue likely stems from numpy's internal string handling treating `'\x00'` as a string terminator (C-style null-terminated strings). The fix would require:

1. Properly handling null characters as valid string content, not terminators
2. Using length-based string handling instead of null-terminated strings
3. Or documenting that numpy.char doesn't support null characters in strings