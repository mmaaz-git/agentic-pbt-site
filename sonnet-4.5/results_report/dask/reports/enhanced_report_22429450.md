# Bug Report: dask.utils.parse_bytes Accepts Negative Values

**Target**: `dask.utils.parse_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_bytes` function accepts negative numeric values and string representations of negative byte sizes, returning negative integers which violates the semantic meaning of byte sizes as non-negative quantities of data.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import parse_bytes


@given(
    st.integers(max_value=-1),
    st.sampled_from(['kB', 'MB', 'GB', 'KiB', 'MiB', 'GiB', 'B', ''])
)
def test_parse_bytes_rejects_negative_strings(n, unit):
    s = f"{n}{unit}"
    result = parse_bytes(s)
    assert result >= 0, f"parse_bytes('{s}') returned negative value {result}"

if __name__ == "__main__":
    test_parse_bytes_rejects_negative_strings()
```

<details>

<summary>
**Failing input**: `n=-1, unit='kB'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 15, in <module>
    test_parse_bytes_rejects_negative_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 6, in test_parse_bytes_rejects_negative_strings
    st.integers(max_value=-1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 12, in test_parse_bytes_rejects_negative_strings
    assert result >= 0, f"parse_bytes('{s}') returned negative value {result}"
           ^^^^^^^^^^^
AssertionError: parse_bytes('-1kB') returned negative value -1000
Falsifying example: test_parse_bytes_rejects_negative_strings(
    # The test always failed when commented parts were varied together.
    n=-1,  # or any other generated value
    unit='kB',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import parse_bytes

# Test negative values with various units
print("Testing parse_bytes with negative values:")
print(f"parse_bytes('-128MiB') = {parse_bytes('-128MiB')}")
print(f"parse_bytes(-100) = {parse_bytes(-100)}")
print(f"parse_bytes('-5kB') = {parse_bytes('-5kB')}")
print(f"parse_bytes('-1B') = {parse_bytes('-1B')}")
print(f"parse_bytes('-1GB') = {parse_bytes('-1GB')}")
print(f"parse_bytes(-1024) = {parse_bytes(-1024)}")

# Show that these negative values are semantically incorrect
print("\nByte sizes should represent amounts of data, which cannot be negative.")
print("Negative byte sizes make no semantic sense - you cannot have -5MB of RAM or -100KB file size.")
```

<details>

<summary>
parse_bytes accepts and returns negative byte sizes
</summary>
```
Testing parse_bytes with negative values:
parse_bytes('-128MiB') = -134217728
parse_bytes(-100) = -100
parse_bytes('-5kB') = -5000
parse_bytes('-1B') = -1
parse_bytes('-1GB') = -1000000000
parse_bytes(-1024) = -1024

Byte sizes should represent amounts of data, which cannot be negative.
Negative byte sizes make no semantic sense - you cannot have -5MB of RAM or -100KB file size.
```
</details>

## Why This Is A Bug

Byte sizes fundamentally represent amounts of data storage or memory, which are inherently non-negative quantities. You cannot have -5MB of RAM, -100KB file size, or -1GB of disk space. The function name `parse_bytes` and its documented purpose of converting "byte string representations to numeric values" implies it should handle valid byte size specifications, which by definition must be non-negative.

The function's docstring (lines 1586-1613 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py`) provides only positive value examples, never demonstrating or discussing negative values. This suggests negative values were not intended to be supported.

More critically, this bug can cause incorrect behavior in downstream code. The `parse_bytes` function is used in `dask.bytes.read_bytes` (lines 90 and 167 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bytes/core.py`) to parse `blocksize` and `sample` parameters. A negative blocksize would break the block calculation logic (lines 124-143), potentially causing division errors or incorrect data partitioning.

## Relevant Context

The function implementation (lines 1585-1639 in `dask/utils.py`) currently:
1. For numeric inputs (int/float), simply returns `int(s)` without validation (line 1615)
2. For string inputs, uses `float(prefix)` which accepts negative numbers (line 1629)
3. Multiplies by the byte size multiplier and returns the result without checking if it's negative (lines 1638-1639)

The byte_sizes dictionary (lines 1642-1658) defines valid unit multipliers but doesn't enforce non-negative values.

Documentation: https://docs.dask.org/en/stable/api.html#dask.utils.parse_bytes

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1612,6 +1612,8 @@ def parse_bytes(s: float | str) -> int:
     ValueError: Could not interpret 'foos' as a byte unit
     """
     if isinstance(s, (int, float)):
+        if s < 0:
+            raise ValueError(f"Byte size cannot be negative: {s}")
         return int(s)
     s = s.replace(" ", "")
     if not any(char.isdigit() for char in s):
@@ -1636,6 +1638,8 @@ def parse_bytes(s: float | str) -> int:
         raise ValueError("Could not interpret '%s' as a byte unit" % suffix) from e

     result = n * multiplier
+    if result < 0:
+        raise ValueError(f"Byte size cannot be negative: {s!r} evaluates to {result}")
     return int(result)
```