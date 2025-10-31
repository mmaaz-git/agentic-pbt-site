# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation for Annual Frequencies

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate their mathematical inverse relationship when comparing annual frequencies to themselves, returning inconsistent results that break the expected symmetry between upsampling and downsampling operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

VALID_FREQUENCIES = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT",
    "M", "MS", "BM", "BMS",
    "Q", "QS", "BQ", "BQS",
    "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "Y", "YS", "BY", "BYS",
    "Y-JAN", "Y-FEB", "Y-MAR", "Y-APR", "Y-MAY", "Y-JUN",
    "Y-JUL", "Y-AUG", "Y-SEP", "Y-OCT", "Y-NOV", "Y-DEC",
]

freq_strategy = st.sampled_from(VALID_FREQUENCIES)

@given(source=freq_strategy, target=freq_strategy)
@settings(max_examples=1000)
def test_inverse_relationship_superperiod_subperiod(source, target):
    """
    Property: If is_superperiod(source, target) is True,
    then is_subperiod(target, source) should also be True.
    """
    super_result = is_superperiod(source, target)
    sub_result = is_subperiod(target, source)

    if super_result:
        assert sub_result, (
            f"is_superperiod({source!r}, {target!r}) = True, "
            f"but is_subperiod({target!r}, {source!r}) = {sub_result}"
        )

if __name__ == "__main__":
    test_inverse_relationship_superperiod_subperiod()
```

<details>

<summary>
**Failing input**: `source='Y-JAN', target='Y-JAN'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 36, in <module>
    test_inverse_relationship_superperiod_subperiod()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 20, in test_inverse_relationship_superperiod_subperiod
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 30, in test_inverse_relationship_superperiod_subperiod
    assert sub_result, (
           ^^^^^^^^^^
AssertionError: is_superperiod('Y-JAN', 'Y-JAN') = True, but is_subperiod('Y-JAN', 'Y-JAN') = False
Falsifying example: test_inverse_relationship_superperiod_subperiod(
    source='Y-JAN',
    target='Y-JAN',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/53/hypo.py:31
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py:512
```
</details>

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq = 'Y-JAN'

super_result = is_superperiod(freq, freq)
sub_result = is_subperiod(freq, freq)

print(f"is_superperiod('{freq}', '{freq}') = {super_result}")
print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")

assert super_result == sub_result, \
    f"Expected both to return the same value, but got {super_result} and {sub_result}"
```

<details>

<summary>
AssertionError: Expected both to return the same value
</summary>
```
is_superperiod('Y-JAN', 'Y-JAN') = True
is_subperiod('Y-JAN', 'Y-JAN') = False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/repo.py", line 11, in <module>
    assert super_result == sub_result, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected both to return the same value, but got True and False
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property of frequency relationships in pandas. The functions `is_superperiod` and `is_subperiod` are documented to check if upsampling or downsampling is "possible" between frequencies. For these functions to be consistent, they must maintain an inverse relationship: if `is_superperiod(A, B)` returns `True`, then `is_subperiod(B, A)` must also return `True`.

When comparing a frequency to itself (identity case), the expected behavior is that both functions should return the same value since neither upsampling nor downsampling is actually needed. The current implementation returns `True` for `is_superperiod('Y-JAN', 'Y-JAN')` but `False` for `is_subperiod('Y-JAN', 'Y-JAN')`, which is logically incoherent.

This bug specifically affects all annual frequencies with month anchors: Y, Y-JAN, Y-FEB, Y-MAR, Y-APR, Y-MAY, Y-JUN, Y-JUL, Y-AUG, Y-SEP, Y-OCT, Y-NOV, Y-DEC. Other frequency types (daily, weekly, hourly, etc.) correctly return consistent values for both functions when comparing to themselves.

## Relevant Context

The root cause is an asymmetry in the implementation at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py`:

- `is_superperiod` (lines 510-512) explicitly checks if both source and target are annual frequencies and returns `True` when their month anchors match
- `is_subperiod` (lines 455-460) lacks this corresponding check for annual-to-annual comparisons, falling through to line 460 which returns `False` for annual targets unless the source is in a specific set of non-annual frequencies

This is clearly an oversight in the implementation where the symmetrical case was handled in one function but not its inverse. The bug affects a core part of pandas' time series functionality that users rely on for frequency conversion checks in data resampling operations.

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -454,6 +454,8 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```