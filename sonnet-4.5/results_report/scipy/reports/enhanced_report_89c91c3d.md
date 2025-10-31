# Bug Report: scipy.odr.ODR.set_iprint Crashes with IndexError and ValueError

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ODR.set_iprint()` method crashes with `IndexError` or `ValueError` when the `iprint` attribute contains values with digits outside the expected encoding range. The method attempts to decode `iprint` digits using a lookup table with only 7 entries, but the extracted digits can range from 0-9.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.odr import ODR, Data, Model
import pytest

def linear_func(B, x):
    return B[0] * x + B[1]

@given(st.integers(min_value=0, max_value=9999))
@settings(max_examples=200)
def test_set_iprint_robust_to_manual_iprint(manual_iprint):
    data = Data([1, 2, 3], [2, 4, 6])
    model = Model(linear_func)
    odr = ODR(data, model, beta0=[1, 1])

    odr.iprint = manual_iprint

    odr.set_iprint(final=0)
```

<details>

<summary>
**Failing input**: `manual_iprint=7` and `manual_iprint=3`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 20, in <module>
  |     test_set_iprint_robust_to_manual_iprint()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 9, in test_set_iprint_robust_to_manual_iprint
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 17, in test_set_iprint_robust_to_manual_iprint
    |     odr.set_iprint(final=0)
    |     ~~~~~~~~~~~~~~^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1060, in set_iprint
    |     iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
    |                                                ~~~~~~^^^^^^^
    | IndexError: list index out of range
    | Falsifying example: test_set_iprint_robust_to_manual_iprint(
    |     manual_iprint=7,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 17, in test_set_iprint_robust_to_manual_iprint
    |     odr.set_iprint(final=0)
    |     ~~~~~~~~~~~~~~^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1081, in set_iprint
    |     ip[3] = ip2arg.index(iprint_l[4:6])
    |             ~~~~~~~~~~~~^^^^^^^^^^^^^^^
    | ValueError: [0, 1] is not in list
    | Falsifying example: test_set_iprint_robust_to_manual_iprint(
    |     manual_iprint=3,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from scipy.odr import ODR, Data, Model

def linear_func(B, x):
    return B[0] * x + B[1]

data = Data([1, 2, 3], [2, 4, 6])
model = Model(linear_func)
odr = ODR(data, model, beta0=[1, 1])

# This should crash with IndexError
odr.iprint = 67
odr.set_iprint(final=0)
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 12, in <module>
    odr.set_iprint(final=0)
    ~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py", line 1060, in set_iprint
    iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
                                               ~~~~~~^^^^^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

The `set_iprint` method has a fundamental design flaw in how it encodes and decodes the `iprint` parameter. The method extracts individual digits from `iprint` (lines 1038-1041 in `_odrpack.py`) where each digit can range from 0-9. However, it then uses these digits to index into `ip2arg`, which only contains 7 entries (indices 0-6).

When `iprint` contains any digit >= 7 (e.g., 7, 17, 67, 70, 170, 777, 8000, 9999), the code crashes with `IndexError` at line 1060. Additionally, certain combinations of valid digits can produce pairs that don't exist in `ip2arg`, causing `ValueError` when the method tries to find them using `ip2arg.index()` at lines 1079-1081.

The ODR constructor accepts `iprint` as a parameter and allows direct attribute assignment, making this a public API. The documentation for `set_iprint` mentions "If iprint is not set manually or with this method" but doesn't explicitly forbid manual setting or document the valid range. Users can reasonably expect that either:
1. Invalid `iprint` values should be rejected when set, or
2. `set_iprint()` should handle or validate pre-existing `iprint` values

Neither safeguard exists, leading to crashes on legitimate API usage patterns.

## Relevant Context

The `iprint` parameter controls printing of computation reports in ODRPACK. It's encoded as a 4-digit integer where each digit controls different aspects of reporting. The `ip2arg` lookup table maps between digit values and [rptfile, stdout] reporting level pairs.

The bug manifests when:
- Loading ODR configurations from files
- Copying settings between ODR instances
- Setting `iprint` based on computed values
- Any scenario where `iprint` is set before calling `set_iprint()`

Link to affected code: https://github.com/scipy/scipy/blob/main/scipy/odr/_odrpack.py#L1009-L1084

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1057,7 +1057,14 @@ class ODR:
             raise OdrError(
                 "no rptfile specified, cannot output to stdout twice")

-        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
+        # Validate that iprint digits are within valid range for ip2arg
+        for idx, digit_pos in enumerate([0, 1, 3]):
+            if not 0 <= ip[digit_pos] < len(ip2arg):
+                raise ValueError(
+                    f"Invalid iprint value {self.iprint}: digit at position "
+                    f"{digit_pos} is {ip[digit_pos]}, but must be in range [0, {len(ip2arg)-1}]"
+                )
+
+        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]

         if init is not None:
             iprint_l[0] = init
@@ -1076,9 +1083,17 @@ class ODR:
             # 0..9
             ip[2] = iter_step

-        ip[0] = ip2arg.index(iprint_l[0:2])
-        ip[1] = ip2arg.index(iprint_l[2:4])
-        ip[3] = ip2arg.index(iprint_l[4:6])
+        try:
+            ip[0] = ip2arg.index(iprint_l[0:2])
+            ip[1] = ip2arg.index(iprint_l[2:4])
+            ip[3] = ip2arg.index(iprint_l[4:6])
+        except ValueError as e:
+            valid_combinations = ", ".join([str(combo) for combo in ip2arg])
+            raise ValueError(
+                f"Invalid iprint parameter combination. The combination "
+                f"{iprint_l[0:2]}, {iprint_l[2:4]}, or {iprint_l[4:6]} is not valid. "
+                f"Valid combinations are: {valid_combinations}"
+            ) from e

         self.iprint = ip[0]*1000 + ip[1]*100 + ip[2]*10 + ip[3]
```