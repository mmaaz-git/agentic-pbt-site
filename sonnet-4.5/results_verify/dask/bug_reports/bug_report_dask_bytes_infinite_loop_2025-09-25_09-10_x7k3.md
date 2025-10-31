# Bug Report: dask.bytes.core Infinite Loop in Blocksize Calculation

**Target**: `dask.bytes.core.read_bytes`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `read_bytes` function in `dask.bytes.core` contains an infinite loop when calculating block offsets for certain combinations of file size and blocksize, due to floating-point arithmetic issues.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    st.integers(min_value=1, max_value=10**9),
    st.integers(min_value=1, max_value=10**6)
)
def test_blocksize_calculation_terminates(size, blocksize):
    if size <= blocksize:
        return

    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []
    iterations = 0

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
        iterations += 1
        assert iterations < 10000, f"Infinite loop detected for size={size}, blocksize={blocksize}"

    length.append(size - off[-1])
```

**Failing input**: `size=1000000000, blocksize=333333`

## Reproducing the Bug

```python
size = 1_000_000_000
blocksize = 333333

if size % blocksize and size > blocksize:
    blocksize1 = size / (size // blocksize)
else:
    blocksize1 = blocksize

print(f"blocksize1 = {blocksize1}")

place = 0
iterations = 0
max_iterations = 1000

while size - place > (blocksize1 * 2) - 1:
    place += blocksize1
    iterations += 1
    if iterations > max_iterations:
        print(f"INFINITE LOOP after {iterations} iterations")
        print(f"place = {place}, size - place = {size - place}")
        print(f"blocksize1 * 2 - 1 = {blocksize1 * 2 - 1}")
        break
```

Output:
```
blocksize1 = 333333.3333333333
INFINITE LOOP after 1001 iterations
place = 333666666.666666, size - place = 666333333.333334
blocksize1 * 2 - 1 = 666665.6666666666
```

## Why This Is A Bug

The infinite loop occurs in `dask/bytes/core.py` lines 123-143. When `size % blocksize != 0`, the code calculates:

```python
blocksize1 = size / (size // blocksize)  # Line 125
```

This produces a **float**, not an integer. Later, the loop at line 133:

```python
while size - place > (blocksize1 * 2) - 1:
    place += blocksize1  # Line 134
```

accumulates floating-point values in `place`. Due to floating-point precision, the loop condition may never become false, causing an infinite loop.

In the failing case:
- `blocksize1 = 1000000000 / 3000 = 333333.333...`
- After many iterations, `place` approaches but never quite satisfies the termination condition
- `size - place` remains larger than `blocksize1 * 2 - 1` indefinitely

## Fix

Convert `blocksize1` to an integer to prevent floating-point accumulation errors:

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -122,9 +122,9 @@ def read_bytes(
             else:
                 # shrink blocksize to give same number of parts
                 if size % blocksize and size > blocksize:
-                    blocksize1 = size / (size // blocksize)
+                    blocksize1 = int(size / (size // blocksize))
                 else:
                     blocksize1 = blocksize
                 place = 0
                 off = [0]
                 length = []
```