# Bug Report: format_bytes Length Guarantee Violated for Large Values

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB (petabytes).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes


@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)}, expected <= 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()
```

<details>

<summary>
**Failing input**: `n=1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 12, in <module>
    test_format_bytes_length_claim()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 6, in test_format_bytes_length_claim
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 9, in test_format_bytes_length_claim
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)}, expected <= 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11, expected <= 10
Falsifying example: test_format_bytes_length_claim(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")

assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
```

<details>

<summary>
AssertionError: Expected <= 10 characters, got 11
</summary>
```
Input: 1125894277343089729
Input < 2**60: True
Output: '1000.00 PiB'
Output length: 11
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/repo.py", line 11, in <module>
    assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
           ^^^^^^^^^^^^^^^^^
AssertionError: Expected <= 10 characters, got 11
```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees that "For all values < 2**60, the output is always <= 10 characters" (line 1788 in dask/utils.py). This is a clear, documented contract that users may rely upon for fixed-width formatting in terminals, aligned table columns, or monitoring dashboards.

The bug occurs when values divided by 2**50 (PiB scale) result in a number >= 1000. The formatting string `f"{n / k:.2f} {prefix}B"` produces outputs like "1000.00 PiB" which are 11 characters long (4 digits + decimal point + 2 decimals + space + 3-char suffix = 11 characters total).

The violation happens for all byte values in the range [1125894277343089729, 1152921504606846975], which correspond to formatted outputs from "1000.00 PiB" to "1024.00 PiB". While these are extremely large values (exabyte scale), the function explicitly claims to handle all values < 2**60 with the 10-character guarantee.

## Relevant Context

- The format_bytes function is located in `/lib/python3.13/site-packages/dask/utils.py` starting at line 1771
- Documentation: https://docs.dask.org/en/stable/api.html#dask.utils.format_bytes
- The function correctly uses binary prefixes (kiB, MiB, GiB, TiB, PiB) rather than decimal prefixes
- The threshold check `n >= k * 0.9` ensures values are formatted at the appropriate scale
- Similar formatting functions in the library (like format_time) do not make specific length guarantees

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1795,7 +1795,11 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
     return f"{n} B"
```