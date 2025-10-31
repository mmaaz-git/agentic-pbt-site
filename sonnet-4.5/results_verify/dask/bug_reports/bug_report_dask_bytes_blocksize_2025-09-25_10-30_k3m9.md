# Bug Report: dask.bytes Blocksize Calculation Quadratic Time Complexity

**Target**: `dask.bytes.core.read_bytes`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The blocksize calculation loop in `read_bytes` has O(size/blocksize) time complexity, causing practical hangs or extremely slow performance when reading large files with small blocksizes.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st, settings
import time


@given(
    st.integers(min_value=1, max_value=10**9),
    st.integers(min_value=1, max_value=10**8)
)
@settings(max_examples=100)
def test_blocksize_calculation_terminates_quickly(size, blocksize):
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    iterations = 0
    start = time.time()

    while size - place > (blocksize1 * 2) - 1:
        iterations += 1
        place += blocksize1

        elapsed = time.time() - start
        if elapsed > 0.1:
            raise AssertionError(
                f"Loop did not terminate in reasonable time. "
                f"size={size}, blocksize={blocksize}, "
                f"iterations={iterations}, elapsed={elapsed:.2f}s"
            )
```

**Failing input**: `size=141335, blocksize=1` (took 141,124 iterations and 0.10s)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

size = 1_000_000_000
blocksize = 1

if size % blocksize and size > blocksize:
    blocksize1 = size / (size // blocksize)
else:
    blocksize1 = blocksize

place = 0
iterations = 0

while size - place > (blocksize1 * 2) - 1:
    iterations += 1
    place += blocksize1

    if iterations == 1_000_000:
        print(f"Already at {iterations:,} iterations - this will take ~500 million total")
        break

print(f"Expected total iterations: ~{size // (blocksize * 2):,}")
```

## Why This Is A Bug

The loop at lines 133-137 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bytes/core.py` increments by `blocksize1` each iteration. When `blocksize` is small relative to `size`, this requires O(size/blocksize) iterations.

For example, with a 1GB file (`size=10**9`) and `blocksize=1`, the loop would run ~500 million iterations, taking minutes to hours to complete. This makes `read_bytes` unusable for large files with small blocksizes, which is a valid use case (e.g., reading newline-delimited files where blocksize could be as small as a few bytes).

## Fix

The loop should calculate offsets arithmetically rather than iteratively. The number of blocks and their positions can be computed directly:

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -122,20 +122,16 @@ def read_bytes(
             else:
                 # shrink blocksize to give same number of parts
                 if size % blocksize and size > blocksize:
                     blocksize1 = size / (size // blocksize)
                 else:
                     blocksize1 = blocksize
-                place = 0
-                off = [0]
-                length = []
-
-                # figure out offsets, spreading around spare bytes
-                while size - place > (blocksize1 * 2) - 1:
-                    place += blocksize1
-                    off.append(int(place))
-                    length.append(off[-1] - off[-2])
-                length.append(size - off[-1])
+
+                # calculate number of blocks and their offsets
+                num_blocks = int(size // blocksize1)
+                off = [int(i * blocksize1) for i in range(num_blocks)]
+                if off[-1] < size:
+                    off.append(size)
+                length = [off[i+1] - off[i] for i in range(len(off)-1)]

                 if not_zero:
                     off[0] = 1
                     length[0] -= 1
```