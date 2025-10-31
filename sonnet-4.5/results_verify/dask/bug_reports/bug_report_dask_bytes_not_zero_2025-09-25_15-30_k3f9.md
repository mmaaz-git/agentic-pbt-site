# Bug Report: dask.bytes.core.read_bytes not_zero Parameter Creates Invalid Blocks

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `not_zero=True` and the first block would normally have length 1, the blocksize calculation produces invalid results: zero-length blocks and duplicate offsets.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume


def compute_offsets_and_lengths(size, blocksize, not_zero=False):
    if size == 0:
        return [], []

    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
    length.append(size - off[-1])

    if not_zero:
        off[0] = 1
        length[0] -= 1

    return off, length


@given(
    st.integers(min_value=2, max_value=10**9),
    st.integers(min_value=1, max_value=10**6),
)
@settings(max_examples=2000, deadline=None)
def test_not_zero_flag(size, blocksize):
    assume(blocksize <= size)
    assume(size / blocksize <= 100000)

    offsets, lengths = compute_offsets_and_lengths(size, blocksize, not_zero=True)

    assert offsets[0] == 1, "With not_zero=True, first offset should be 1"
    assert all(length > 0 for length in lengths), "All lengths must be positive even with not_zero"

    total_bytes = sum(lengths)
    assert total_bytes == size - 1, f"With not_zero, total bytes should be {size - 1}, got {total_bytes}"
```

**Failing input**: `size=2, blocksize=1`

## Reproducing the Bug

```python
def compute_offsets_and_lengths(size, blocksize, not_zero=False):
    if size == 0:
        return [], []

    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
    length.append(size - off[-1])

    if not_zero:
        off[0] = 1
        length[0] -= 1

    return off, length


size, blocksize = 2, 1
offsets, lengths = compute_offsets_and_lengths(size, blocksize, not_zero=True)
print(f"Offsets: {offsets}")
print(f"Lengths: {lengths}")

assert all(l > 0 for l in lengths), f"Found zero/negative length: {lengths}"
```

Expected behavior: All block lengths should be positive.

Actual output:
```
Offsets: [1, 1]
Lengths: [0, 1]
AssertionError: Found zero/negative length: [0, 1]
```

## Why This Is A Bug

The `not_zero` parameter is documented to "Force seek of start-of-file delimiter, discarding header." However, when the first block would normally be 1 byte, the code:

1. Sets `off[0] = 1` (changing it from 0)
2. Decrements `length[0]` by 1

This causes `length[0]` to become 0, violating the invariant that all block lengths must be positive. Additionally, `off[0]` can now equal `off[1]`, creating duplicate offsets.

This would cause downstream code that reads these blocks to fail or behave incorrectly.

## Fix

The bug occurs because the code unconditionally decrements `length[0]` without checking if it would become zero or negative. The fix should ensure the first block length remains positive after adjustment:

```diff
diff --git a/dask/bytes/core.py b/dask/bytes/core.py
index 1234567..abcdefg 100644
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -137,8 +137,11 @@ def read_bytes(
                 length.append(size - off[-1])

                 if not_zero:
-                    off[0] = 1
-                    length[0] -= 1
+                    if length[0] > 1:
+                        off[0] = 1
+                        length[0] -= 1
+                    else:
+                        raise ValueError("Cannot use not_zero=True when first block is 1 byte or less")
                 offsets.append(off)
                 lengths.append(length)
```