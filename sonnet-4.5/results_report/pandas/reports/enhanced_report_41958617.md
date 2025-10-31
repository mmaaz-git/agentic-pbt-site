# Bug Report: pandas.io.formats.format.format_percentiles Violates Uniqueness Contract for Denormalized Floats

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented contract that unique inputs remain unique after rounding when given denormalized floating-point values extremely close to zero (e.g., `[0.0, 2.4505010068371657e-250]`), producing duplicate outputs `['0%', '0%']`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20, unique=True))
@settings(max_examples=1000)
def test_format_percentiles_unique_inputs_remain_unique(percentiles):
    """
    Property from docstring: "if any two elements of percentiles differ,
    they remain different after rounding"
    """
    formatted = format_percentiles(percentiles)
    assert len(formatted) == len(set(formatted)), \
        f"Unique inputs produced duplicate outputs: {percentiles} -> {formatted}"

if __name__ == "__main__":
    test_format_percentiles_unique_inputs_remain_unique()
    print("All tests passed!")
```

<details>

<summary>
**Failing input**: `[0.0, 2.4505010068371657e-250]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 16, in <module>
    test_format_percentiles_unique_inputs_remain_unique()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_format_percentiles_unique_inputs_remain_unique
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 12, in test_format_percentiles_unique_inputs_remain_unique
    assert len(formatted) == len(set(formatted)), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Unique inputs produced duplicate outputs: [0.0, 2.4505010068371657e-250] -> ['0%', '0%']
Falsifying example: test_format_percentiles_unique_inputs_remain_unique(
    percentiles=[0.0, 2.4505010068371657e-250],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/56/hypo.py:13
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

# Test case with extremely small value close to zero
percentiles = [0.0, 7.506590166045388e-253]

# Format the percentiles
formatted = format_percentiles(percentiles)

# Display results
print(f"Input percentiles: {percentiles}")
print(f"Unique input count: {len(set(percentiles))}")
print(f"Output formatted: {formatted}")
print(f"Unique output count: {len(set(formatted))}")

# Check if uniqueness is preserved
try:
    assert len(formatted) == len(set(formatted)), \
        f"Unique inputs should produce unique outputs, but got {formatted}"
    print("\nAssertion passed: Uniqueness preserved")
except AssertionError as e:
    print(f"\nAssertion failed: {e}")
```

<details>

<summary>
AssertionError: Unique inputs should produce unique outputs
</summary>
```
Input percentiles: [0.0, 7.506590166045388e-253]
Unique input count: 2
Output formatted: ['0%', '0%']
Unique output count: 1

Assertion failed: Unique inputs should produce unique outputs, but got ['0%', '0%']
```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees in its Notes section (lines 1562-1563): "Rounding precision is chosen so that: (1) if any two elements of ``percentiles`` differ, they remain different after rounding". This is an unqualified, absolute guarantee with no exceptions mentioned for denormalized floats or extremely small values.

The bug occurs because the function has an early return optimization at lines 1596-1598 when all percentiles are considered "close to integers" via `np.isclose()`. For denormalized floats near zero (values smaller than approximately 2.225e-308), both `0.0` and the tiny value round to integer 0 and satisfy the `isclose` check. This early return path bypasses the uniqueness-preserving logic at lines 1600-1601 that recalculates precision based on `np.unique(percentiles)`, which would detect and preserve the distinction between the values.

The documented contract makes no exceptions for "practical" values or denormalized floats - it promises that ANY different values will remain different after formatting.

## Relevant Context

- This is an internal utility function (not public API) used by pandas' `describe()` method to format percentile labels
- The failing values are denormalized floating-point numbers (subnormal numbers) in the range of e-250 to e-253
- The function correctly handles the uniqueness guarantee for normal floating-point values but fails for subnormal values very close to zero
- Source code location: `/pandas/io/formats/format.py`, lines 1546-1606
- The `get_precision` function (lines 1609-1616) calculates the appropriate decimal precision to maintain distinctions between values

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1589,17 +1589,10 @@ def format_percentiles(
         raise ValueError("percentiles should all be in the interval [0,1]")

     percentiles = 100 * percentiles
-    prec = get_precision(percentiles)
-    percentiles_round_type = percentiles.round(prec).astype(int)
-
-    int_idx = np.isclose(percentiles_round_type, percentiles)
-
-    if np.all(int_idx):
-        out = percentiles_round_type.astype(str)
-        return [i + "%" for i in out]
-
     unique_pcts = np.unique(percentiles)
     prec = get_precision(unique_pcts)
+    percentiles_round_type = percentiles.round(prec).astype(int)
+    int_idx = np.isclose(percentiles_round_type, percentiles)
     out = np.empty_like(percentiles, dtype=object)
     out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
```