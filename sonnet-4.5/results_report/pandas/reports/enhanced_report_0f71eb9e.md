# Bug Report: pandas.tseries.frequencies Symmetric Relationship Violation in is_superperiod/is_subperiod

**Target**: `pandas.tseries.frequencies.is_superperiod` and `pandas.tseries.frequencies.is_subperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_superperiod` and `is_subperiod` functions violate their expected symmetry property for business day ('B') and daily ('D') frequencies, incorrectly allowing both to be superperiods of each other.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq_module
from hypothesis import given, strategies as st, settings, example

PERIOD_FREQUENCIES = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'B', 'BM', 'BQ', 'BY']

@given(
    source=st.sampled_from(PERIOD_FREQUENCIES),
    target=st.sampled_from(PERIOD_FREQUENCIES),
)
@example(source='D', target='B')
@settings(max_examples=200)
def test_is_subperiod_superperiod_symmetry(source, target):
    is_sub = freq_module.is_subperiod(source, target)
    is_super = freq_module.is_superperiod(target, source)

    assert is_sub == is_super, (
        f"Symmetry violated: is_subperiod({source}, {target})={is_sub} but "
        f"is_superperiod({target}, {source})={is_super}"
    )

if __name__ == "__main__":
    test_is_subperiod_superperiod_symmetry()
```

<details>

<summary>
**Failing input**: `source='D', target='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 22, in <module>
    test_is_subperiod_superperiod_symmetry()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_is_subperiod_superperiod_symmetry
    source=st.sampled_from(PERIOD_FREQUENCIES),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 16, in test_is_subperiod_superperiod_symmetry
    assert is_sub == is_super, (
           ^^^^^^^^^^^^^^^^^^
AssertionError: Symmetry violated: is_subperiod(D, B)=False but is_superperiod(B, D)=True
Falsifying explicit example: test_is_subperiod_superperiod_symmetry(
    source='D',
    target='B',
)
```
</details>

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq_module

source, target = 'D', 'B'

is_sub_dt = freq_module.is_subperiod(source, target)
is_super_dt = freq_module.is_superperiod(source, target)
is_sub_td = freq_module.is_subperiod(target, source)
is_super_td = freq_module.is_superperiod(target, source)

print(f"is_subperiod('D', 'B') = {is_sub_dt}")
print(f"is_superperiod('D', 'B') = {is_super_dt}")
print(f"is_subperiod('B', 'D') = {is_sub_td}")
print(f"is_superperiod('B', 'D') = {is_super_td}")

assert not (is_super_dt and is_super_td), "Both cannot be superperiods of each other!"
```

<details>

<summary>
Logical impossibility: Both frequencies claim to be superperiods
</summary>
```
is_subperiod('D', 'B') = False
is_superperiod('D', 'B') = True
is_subperiod('B', 'D') = False
is_superperiod('B', 'D') = True
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 15, in <module>
    assert not (is_super_dt and is_super_td), "Both cannot be superperiods of each other!"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Both cannot be superperiods of each other!
```
</details>

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` are meant to establish a hierarchical relationship between time frequencies, where:

- `is_subperiod(source, target)` returns True if you can downsample from `source` to `target` (source has higher frequency than target)
- `is_superperiod(source, target)` returns True if you can upsample from `source` to `target` (source has lower frequency than target)

These functions must satisfy a fundamental symmetry property: **if `is_subperiod(A, B)` is True, then `is_superperiod(B, A)` must also be True**, and vice versa. This represents the same hierarchical relationship viewed from opposite directions.

However, the current implementation violates this property for business days ('B') and regular days ('D'):

1. `is_subperiod('D', 'B')` returns False - claims you cannot downsample from daily to business daily
2. `is_superperiod('D', 'B')` returns True - claims you can upsample from daily to business daily
3. `is_subperiod('B', 'D')` returns False - claims you cannot downsample from business daily to daily
4. `is_superperiod('B', 'D')` returns True - claims you can upsample from business daily to daily

This creates a logical impossibility where both frequencies are simultaneously "larger" than each other in the hierarchy. The symmetry violation means: `is_subperiod('D', 'B') = False` but `is_superperiod('B', 'D') = True`, which contradicts the expected relationship.

## Relevant Context

Looking at the source code in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py`:

- Lines 467-468 handle 'B' as target in `is_subperiod`: only allows specific sources, excluding 'D'
- Lines 525-530 handle 'B' and 'D' as sources in `is_superperiod`: both incorrectly include the other as a valid target

The issue stems from how business day frequencies are treated. Business days are a subset of calendar days (excludes weekends), making them fundamentally incomparable in a strict frequency hierarchy. They represent different sampling patterns rather than different frequencies.

Documentation link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.tseries.frequencies.is_superperiod.html

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -523,9 +523,9 @@ def is_superperiod(source, target) -> bool:
     elif _is_weekly(source):
         return target in {source, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif source == "B":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"B", "h", "min", "s", "ms", "us", "ns"}
     elif source == "C":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"C", "h", "min", "s", "ms", "us", "ns"}
     elif source == "D":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"D", "h", "min", "s", "ms", "us", "ns"}
     elif source == "h":
         return target in {"h", "min", "s", "ms", "us", "ns"}
```