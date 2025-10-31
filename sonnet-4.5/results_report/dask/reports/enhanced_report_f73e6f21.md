# Bug Report: dask.utils.format_bytes Violates 10-Character Length Guarantee

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented contract that "For all values < 2**60, the output is always <= 10 characters" by producing an 11-character output for certain edge case values near unit boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"

if __name__ == "__main__":
    # Run the test
    test_format_bytes_length_invariant()
```

<details>

<summary>
**Failing input**: `n = 1125894277343089729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 11, in <module>
    test_format_bytes_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 5, in test_format_bytes_length_invariant
    def test_format_bytes_length_invariant(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 7, in test_format_bytes_length_invariant
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_length_invariant(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print(f"Is n < 2**60? {n < 2**60}")
print(f"2**60 = {2**60}")

# This should pass according to the docstring guarantee
assert len(result) <= 10, f"Length {len(result)} exceeds guaranteed maximum of 10 characters"
```

<details>

<summary>
AssertionError: Length 11 exceeds guaranteed maximum of 10 characters
</summary>
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11
Is n < 2**60? True
2**60 = 1152921504606846976
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/repo.py", line 11, in <module>
    assert len(result) <= 10, f"Length {len(result)} exceeds guaranteed maximum of 10 characters"
           ^^^^^^^^^^^^^^^^^
AssertionError: Length 11 exceeds guaranteed maximum of 10 characters
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 explicitly guarantees: "For all values < 2**60, the output is always <= 10 characters." This is a documented contract that users may rely on for formatting consistency, especially in tabular displays or fixed-width output contexts.

The failing value `1125894277343089729` is definitively less than `2**60` (which equals `1152921504606846976`), yet the function produces '1000.00 PiB' - an 11-character string. This violates the documented guarantee.

The issue occurs because the function uses a threshold of `k * 0.9` (line 1797) to determine when to switch to a larger unit. For PiB (Pebibytes), where k = 2**50, values around 0.9765625 * 2**50 (approximately 1125899906842624 bytes) get formatted with the PiB suffix. When such values are divided by 2**50, they produce results very close to 1000, which format as "1000.00" with 2 decimal places, plus the 4-character suffix " PiB", resulting in 11 total characters.

## Relevant Context

The `format_bytes` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py` starting at line 1771. The function uses binary prefixes (kiB, MiB, GiB, TiB, PiB) and is designed to provide human-readable representations of byte values.

The edge case occurs specifically at the boundary where values are >= 0.9 * 2**50 but < 2**50, causing them to format in PiB units with a value of approximately 1000.00. Similar edge cases theoretically exist for other units (TiB, GiB, MiB, kiB) but only the PiB case violates the 10-character limit due to the specific combination of prefix length and numeric formatting.

Documentation reference: The guarantee is part of the function's docstring and would be visible to users via help(format_bytes) or in API documentation.

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1794,7 +1794,7 @@ def format_bytes(n: int) -> str:
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
-        if n >= k * 0.9:
+        if n >= k:
             return f"{n / k:.2f} {prefix}B"
     return f"{n} B"
```