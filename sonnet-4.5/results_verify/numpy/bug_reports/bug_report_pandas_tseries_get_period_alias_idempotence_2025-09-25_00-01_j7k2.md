# Bug Report: pandas.tseries.frequencies.get_period_alias Idempotence Violation

**Target**: `pandas.tseries.frequencies.get_period_alias`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `get_period_alias` violates the idempotence property: applying the function twice should yield the same result as applying it once (i.e., `f(f(x)) == f(x)`). However, when 'MS', 'QS', 'BQE', or 'BQS' are passed, the function returns 'M' or 'Q', but applying it again to 'M' or 'Q' returns None instead of 'M' or 'Q'.

## Property-Based Test

```python
import pandas.tseries.frequencies
from hypothesis import given, strategies as st, settings, assume

OFFSET_STRINGS = [
    'D', 'B', 'C', 'W', 'M', 'Q', 'Y',
    'BQ', 'BA', 'BM', 'BH', 'BQE', 'BQS', 'BYE', 'BYS',
    'MS', 'ME', 'QS', 'QE', 'YS', 'YE',
    'h', 'min', 's', 'ms', 'us', 'ns',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
]

@given(offset_str=st.sampled_from(OFFSET_STRINGS))
@settings(max_examples=500)
def test_get_period_alias_idempotence(offset_str):
    first_alias = pandas.tseries.frequencies.get_period_alias(offset_str)
    assume(first_alias is not None)

    second_alias = pandas.tseries.frequencies.get_period_alias(first_alias)

    assert second_alias == first_alias, (
        f"get_period_alias('{offset_str}') = '{first_alias}', "
        f"but get_period_alias('{first_alias}') = '{second_alias}'. "
        f"Expected idempotence: f(f(x)) should equal f(x)"
    )
```

**Failing inputs**:
- `'BQE'` -> returns 'Q', then 'Q' -> returns None
- `'MS'` -> returns 'M', then 'M' -> returns None
- `'QS'` -> returns 'Q', then 'Q' -> returns None
- `'BQS'` -> returns 'Q', then 'Q' -> returns None

## Reproducing the Bug

```python
import pandas.tseries.frequencies

examples = ['MS', 'QS', 'BQE', 'BQS']

for offset_str in examples:
    first = pandas.tseries.frequencies.get_period_alias(offset_str)
    second = pandas.tseries.frequencies.get_period_alias(first)
    print(f"get_period_alias('{offset_str}') = '{first}'")
    print(f"get_period_alias('{first}') = '{second}'")
    if second != first:
        print(f"  Idempotence violated!")
    print()
```

## Why This Is A Bug

The function `get_period_alias` is documented to return "Alias to closest period strings BQ->Q etc." The purpose is to normalize offset strings to their canonical period representation. For such a normalization function, idempotence is a critical property: once normalized, applying the function again should not change the result.

The root cause is that `OFFSET_TO_PERIOD_FREQSTR` dictionary (in `pandas/_libs/tslibs/dtypes.pyx`) is missing entries for the canonical period strings themselves:
- It maps 'MS' -> 'M', but doesn't have an entry for 'M' -> 'M'
- It maps 'QS', 'BQE', 'BQS' -> 'Q', but doesn't have an entry for 'Q' -> 'Q'

This means the function is not truly normalizing to a canonical form, because the "canonical form" itself is not in the mapping.

## Fix

The `OFFSET_TO_PERIOD_FREQSTR` dictionary in `pandas/_libs/tslibs/dtypes.pyx` should include entries for the canonical period strings to map to themselves. Add these entries to ensure idempotence:

```diff
--- a/pandas/_libs/tslibs/dtypes.pyx
+++ b/pandas/_libs/tslibs/dtypes.pyx
@@ -xxx,xx +xxx,xx @@ OFFSET_TO_PERIOD_FREQSTR = {
     "D": "D",
     "B": "B",
+    "M": "M",
+    "Q": "Q",
     "min": "min",
     "s": "s",
```