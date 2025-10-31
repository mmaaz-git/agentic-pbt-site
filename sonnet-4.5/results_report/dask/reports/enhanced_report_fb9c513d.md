# Bug Report: dask.utils.format_bytes Output Exceeds 10-Character Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented contract that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB, producing 11-character strings instead.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import dask.utils

@given(st.integers(min_value=0, max_value=2**60))
@settings(max_examples=500)
def test_format_bytes_length_claim(n):
    formatted = dask.utils.format_bytes(n)
    assert len(formatted) <= 10, f"format_bytes({n}) = {formatted!r} has length {len(formatted)} > 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()
```

<details>

<summary>
**Failing input**: `1125894277343089729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 14, in <module>
    test_format_bytes_length_claim()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 8, in test_format_bytes_length_claim
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 11, in test_format_bytes_length_claim
    assert len(formatted) <= 10, f"format_bytes({n}) = {formatted!r} has length {len(formatted)} > 10"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_length_claim(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.utils

# Test the specific failing value
n = 1125899906842624000
result = dask.utils.format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print(f"Is n < 2**60? {n < 2**60} (2**60 = {2**60})")
print()

# Test additional examples to show the boundary
test_cases = [
    1124774006935781376,  # 999 PiB
    1125899906842624000,  # 1000 PiB
    1151795604700004352,  # 1023 PiB
]

print("Additional test cases:")
for test_n in test_cases:
    test_result = dask.utils.format_bytes(test_n)
    print(f"  format_bytes({test_n}) = {test_result!r} (length: {len(test_result)})")
```

<details>

<summary>
Output shows 11-character strings for values >= 1000 PiB
</summary>
```
format_bytes(1125899906842624000) = '1000.00 PiB'
Length: 11
Is n < 2**60? True (2**60 = 1152921504606846976)

Additional test cases:
  format_bytes(1124774006935781376) = '999.00 PiB' (length: 10)
  format_bytes(1125899906842624000) = '1000.00 PiB' (length: 11)
  format_bytes(1151795604700004352) = '1023.00 PiB' (length: 11)
```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees on line 1788 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py`: "For all values < 2**60, the output is always <= 10 characters."

This guarantee is violated when the function formats values >= 1000 PiB (approximately 1.126 * 10^18 bytes). These values are well below 2**60 (1.153 * 10^18), yet produce 11-character outputs like '1000.00 PiB' instead of the promised maximum of 10 characters. The issue occurs because the function uses a fixed 2 decimal place format (`.2f`) which, combined with 4-digit mantissas (1000-1023) and the ' PiB' suffix, results in 11 total characters.

## Relevant Context

The `format_bytes` function currently only supports units up to PiB (pebibyte), iterating through the prefixes ["Pi", "Ti", "Gi", "Mi", "ki"] and using the format string `f"{n / k:.2f} {prefix}B"`. The function switches to a unit when the value is >= 90% of that unit's threshold (`n >= k * 0.9`).

For PiB, this means values from 0.9 * 2^50 bytes up to effectively infinity are formatted as PiB, since there's no larger unit defined. When the mantissa reaches 1000 or higher (which happens at exactly 1024 * 0.9765625 * 2^50 bytes), the output exceeds 10 characters.

Documentation: The guarantee appears both in the source code docstring and in the official Dask API documentation.

## Proposed Fix

Add support for EiB (exbibyte) to handle values >= 1024 PiB, maintaining the 10-character limit:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1790,6 +1790,7 @@ def format_bytes(n: int) -> str:
     for prefix, k in (
+        ("Ei", 2**60),
         ("Pi", 2**50),
         ("Ti", 2**40),
         ("Gi", 2**30),
         ("Mi", 2**20),
```