# Bug Report: dask.utils.format_bytes Exceeds Documented 10-Character Output Bound

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" by producing 11-character outputs for large values near 2**60.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_bound(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

# Run the test
if __name__ == "__main__":
    test_format_bytes_length_bound()
```

<details>

<summary>
**Failing input**: `n = 1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 11, in <module>
    test_format_bytes_length_bound()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 5, in test_format_bytes_length_bound
    def test_format_bytes_length_bound(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 7, in test_format_bytes_length_bound
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_length_bound(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"format_bytes(n) = '{result}'")
print(f"len(result) = {len(result)}")

# Additional boundary tests
print("\n--- Additional tests around the boundary ---")
boundary = 1000 * 2**50
for offset in [-1, 0, 1]:
    test_n = boundary + offset
    test_result = format_bytes(test_n)
    print(f"n = {test_n:,}")
    print(f"format_bytes(n) = '{test_result}'")
    print(f"len(result) = {len(test_result)}")
    print()
```

<details>

<summary>
Output demonstrating 11-character result violating documented 10-character bound
</summary>
```
n = 1125894277343089729
n < 2**60 = True
format_bytes(n) = '1000.00 PiB'
len(result) = 11

--- Additional tests around the boundary ---
n = 1,125,899,906,842,623,999
format_bytes(n) = '1000.00 PiB'
len(result) = 11

n = 1,125,899,906,842,624,000
format_bytes(n) = '1000.00 PiB'
len(result) = 11

n = 1,125,899,906,842,624,001
format_bytes(n) = '1000.00 PiB'
len(result) = 11
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py` explicitly guarantees: "For all values < 2**60, the output is always <= 10 characters." This is a clear contract that the function promises to uphold.

The bug occurs because the function uses the format string `f"{n / k:.2f} {prefix}B"` at line 1798. When formatting values in PiB (pebibytes) where `n / 2**50 >= 1000`, this produces strings in the format "XXXX.XX PiB" which are 11 characters long. Specifically:
- Values where `n >= 1000 * 2**50 = 1,125,899,906,842,624,000` trigger this bug
- These values are still less than 2**60 (1,152,921,504,606,846,976)
- The output "1000.00 PiB" has exactly 11 characters, violating the documented 10-character maximum

This is a contract violation that could affect code relying on the documented guarantee for:
- Fixed-width terminal or table displays
- UI layout calculations
- Buffer size allocations
- Any code that assumes the output will fit in a 10-character field

## Relevant Context

The `format_bytes` function is located in `dask/utils.py` starting at line 1771. It iterates through binary prefixes (PiB, TiB, GiB, MiB, kiB) and formats the value using 2 decimal places when it finds an appropriate scale.

The function works correctly for all practical purposes - it accurately formats byte values into human-readable strings. The issue is purely that it violates its own documented constraint for extremely large values (>= 1000 PiB).

Documentation reference: https://docs.dask.org/en/stable/api.html#dask.utils.format_bytes

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1785,7 +1785,8 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values, the output is <= 10 characters, though large values
+    near 2**60 may produce up to 11 characters (e.g., "1000.00 PiB").
     """
     for prefix, k in (
         ("Pi", 2**50),
```