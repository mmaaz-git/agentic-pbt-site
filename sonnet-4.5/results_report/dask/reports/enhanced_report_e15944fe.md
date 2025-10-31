# Bug Report: dask.utils.format_bytes Output Length Violates Documented 10-Character Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" by producing 11-character outputs for values >= 1000 PiB but < 2**60.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_max_length_10(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

test_format_bytes_max_length_10()
```

<details>

<summary>
**Failing input**: `n=1125894277343089729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 10, in <module>
    test_format_bytes_max_length_10()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 5, in test_format_bytes_max_length_10
    @given(st.integers(min_value=0, max_value=2**60 - 1))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 8, in test_format_bytes_max_length_10
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11, expected <= 10
Falsifying example: test_format_bytes_max_length_10(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"Input: {n} (< 2**60: {n < 2**60})")
print(f"Output: '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Bug: {len(result)} > 10")
```

<details>

<summary>
Output demonstrates 11-character result exceeding documented 10-character limit
</summary>
```
Input: 1125894277343089729 (< 2**60: True)
Output: '1000.00 PiB'
Length: 11 characters
Expected: <= 10 characters
Bug: 11 > 10
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is an unqualified, absolute guarantee about output length.

The bug occurs when formatting values >= 1000 PiB. The format string `f"{n / k:.2f} {prefix}B"` produces:
- 4 digits (1000-1023)
- 1 decimal point
- 2 decimal places
- 1 space
- 3-character prefix ("PiB")
Total: 11 characters

Since 1000 PiB (1,125,899,906,842,624 bytes) is significantly less than 2**60 (1,152,921,504,606,846,976 bytes), these values fall within the documented guarantee but violate it.

## Relevant Context

The function uses a 0.9 threshold to determine unit transitions (line 1797: `if n >= k * 0.9`), meaning values switch to PiB at approximately 900 PiB. The range from 1000 PiB to just under 1024 PiB (which equals 2**60) all produce 11-character outputs.

While these values are astronomically large (1000 PiB ≈ 1.1 exabytes), they are still within the specified input range. Some applications in distributed computing, scientific data processing, or storage management might rely on the documented length guarantee for formatting purposes (e.g., column alignment in tables).

Documentation link: The function is part of the dask.utils module, with source at:
https://github.com/dask/dask/blob/main/dask/utils.py#L1771-L1799

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1795,7 +1795,10 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                return f"{value:.1f} {prefix}B"
+            return f"{value:.2f} {prefix}B"
     return f"{n} B"
```