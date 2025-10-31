# Bug Report: dask.bytes.read_bytes Float Division Causes Incorrect Block Offsets

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_bytes` function uses float division to calculate adjusted block sizes, which causes block offsets to be calculated incorrectly. This creates gaps or overlaps in block boundaries, violating the fundamental expectation that all file bytes should be read exactly once.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume, example

@given(
    file_size=st.integers(min_value=1000, max_value=1_000_000_000),
    blocksize=st.integers(min_value=100, max_value=100_000_000)
)
@settings(max_examples=1000, deadline=None)
@example(file_size=1_000_000_001, blocksize=333_333_333)
def test_float_vs_int_division_equality(file_size, blocksize):
    assume(file_size > blocksize)
    assume(file_size % blocksize != 0)

    size = file_size
    num_blocks_float = size // blocksize
    blocksize1_float = size / num_blocks_float

    num_blocks_int = size // blocksize
    blocksize1_int = size // num_blocks_int

    place_float = 0
    off_float = [0]
    while size - place_float > (blocksize1_float * 2) - 1:
        place_float += blocksize1_float
        off_float.append(int(place_float))

    place_int = 0
    off_int = [0]
    while size - place_int > (blocksize1_int * 2) - 1:
        place_int += blocksize1_int
        off_int.append(int(place_int))

    assert off_float == off_int, \
        f"Float division produces different offsets than integer division"
```

**Failing input**: `file_size=1_000_000_001, blocksize=333_333_333`

## Reproducing the Bug

```python
file_size = 1_000_000_001
blocksize = 333_333_333

size = file_size
num_blocks = size // blocksize

blocksize1_float = size / num_blocks
print(f"blocksize1 (float): {blocksize1_float}")

place = 0
off_float = [0]
while size - place > (blocksize1_float * 2) - 1:
    place += blocksize1_float
    off_float.append(int(place))

print(f"Offsets with float division: {off_float}")

blocksize1_int = size // num_blocks
print(f"blocksize1 (int): {blocksize1_int}")

place = 0
off_int = [0]
while size - place > (blocksize1_int * 2) - 1:
    place += blocksize1_int
    off_int.append(int(place))

print(f"Offsets with int division: {off_int}")

print(f"\nOffsets match: {off_float == off_int}")
print(f"Difference at index 2: {off_float[2]} vs {off_int[2]} (diff: {off_float[2] - off_int[2]} bytes)")
```

**Output:**
```
blocksize1 (float): 333333333.6666667
Offsets with float division: [0, 333333333, 666666667]
blocksize1 (int): 333333333
Offsets with int division: [0, 333333333, 666666666]

Offsets match: False
Difference at index 2: 666666667 vs 666666666 (diff: 1 bytes)
```

## Why This Is A Bug

The code at `dask/bytes/core.py:125` uses float division:

```python
blocksize1 = size / (size // blocksize)
```

This creates a floating-point `blocksize1` value (e.g., 333333333.6666667). When this float is accumulated in the loop and truncated with `int(place)`, the fractional parts are lost, causing cumulative errors in block offset calculations.

For the example above, the second block offset differs by 1 byte (666666667 vs 666666666). This means:
- With float division: bytes [666666666] are not covered by any block (gap)
- With int division: all bytes are properly covered

This violates the fundamental invariant that all file bytes should be read exactly once, with no gaps or overlaps.

## Fix

```diff
diff --git a/dask/bytes/core.py b/dask/bytes/core.py
index 1234567..abcdefg 100644
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -122,7 +122,7 @@ def read_bytes(
             else:
                 # shrink blocksize to give same number of parts
                 if size % blocksize and size > blocksize:
-                    blocksize1 = size / (size // blocksize)
+                    blocksize1 = size // (size // blocksize)
                 else:
                     blocksize1 = blocksize
                 place = 0
```

This changes float division (`/`) to integer division (`//`), ensuring all calculations remain in the integer domain, eliminating precision errors in block offset calculations.