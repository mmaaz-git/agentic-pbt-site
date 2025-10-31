# Bug Report: pandas.tseries.frequencies get_period_alias Not Idempotent

**Target**: `pandas.tseries.frequencies.get_period_alias`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`get_period_alias` is not idempotent. When called on 'ME' it returns 'M', but calling it again on 'M' returns None instead of 'M'. This violates the expected idempotence property that alias functions should have.

## Property-Based Test

```python
from pandas.tseries import frequencies
from hypothesis import given, strategies as st, settings

simple_freq_strings = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'B', 'ME', 'QE', 'YE']

@given(st.sampled_from(simple_freq_strings))
@settings(max_examples=50)
def test_period_alias_consistent(offset_str):
    alias = frequencies.get_period_alias(offset_str)
    if alias is not None:
        alias2 = frequencies.get_period_alias(alias)
        assert alias == alias2, f"get_period_alias not idempotent: {offset_str} -> {alias} -> {alias2}"
```

**Failing input**: `offset_str='ME'`

## Reproducing the Bug

```python
from pandas.tseries import frequencies

alias1 = frequencies.get_period_alias('ME')
alias2 = frequencies.get_period_alias(alias1) if alias1 else None

print(f"get_period_alias('ME') = {repr(alias1)}")
print(f"get_period_alias({repr(alias1)}) = {repr(alias2)}")

assert alias1 == alias2, f"Expected idempotent, got {repr(alias1)} -> {repr(alias2)}"
```

## Why This Is A Bug

The function is documented as "Alias to closest period strings BQ->Q etc." Alias functions should be idempotent - applying the function multiple times should give the same result. If 'ME' aliases to 'M', then 'M' should alias to itself (returning 'M').

The root cause is in the `OFFSET_TO_PERIOD_FREQSTR` dictionary which maps:
- 'ME' -> 'M'
- 'QE' -> 'Q'
- 'YE' -> 'Y'

But 'M' and 'Q' (without 'E' suffix) are not keys in the dictionary, so they return None.

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -125,8 +125,10 @@ OFFSET_TO_PERIOD_FREQSTR: dict[str, str] = {
     "EOM": "M",
     "ME": "M",
     "MS": "M",
+    "M": "M",
     "QE": "Q",
     "QS": "Q",
+    "Q": "Q",
     "BQE": "Q",
     "BQS": "Q",
     "W": "W",
```