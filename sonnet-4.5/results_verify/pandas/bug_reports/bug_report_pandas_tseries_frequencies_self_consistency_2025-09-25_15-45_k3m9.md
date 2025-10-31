# Bug Report: pandas.tseries.frequencies Year Frequency Self-Consistency Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

For the year frequency 'Y', `is_subperiod('Y', 'Y')` returns `False` while `is_superperiod('Y', 'Y')` returns `True`. These functions should return the same value for a frequency compared to itself, as the subperiod and superperiod relationships should be symmetric for self-comparison.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from(['D', 'W', 'M', 'Q', 'Y', 'h', 'B'])

@given(freq_strings)
@settings(max_examples=50)
def test_is_subperiod_self_consistency(freq):
    result_sub = is_subperiod(freq, freq)
    result_super = is_superperiod(freq, freq)

    assert result_sub == result_super, \
        f"Self-inconsistent: is_subperiod({freq}, {freq})={result_sub} != is_superperiod({freq}, {freq})={result_super}"
```

**Failing input**: `freq='Y'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

sub = is_subperiod('Y', 'Y')
sup = is_superperiod('Y', 'Y')

print(f"is_subperiod('Y', 'Y') = {sub}")
print(f"is_superperiod('Y', 'Y') = {sup}")
assert sub == sup
```

Output:
```
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
AssertionError
```

## Why This Is A Bug

This violates the fundamental mathematical property that should hold for subperiod/superperiod relationships:

1. **Reflexive Property**: For any frequency F, the relationship "F is a subperiod of F" should have the same truth value as "F is a superperiod of F". Mathematically, if we define ⊆ as the subperiod relation, then F ⊆ F should equal F ⊇ F.

2. **Logical Consistency**: When comparing a frequency to itself:
   - If the functions use a "strict" interpretation (F ⊂ F is false), both should return `False`
   - If the functions use a "non-strict" interpretation (F ⊆ F is true), both should return `True`
   - Having one return `True` and the other return `False` is logically inconsistent

3. **Other frequencies behave correctly**: Testing shows that D, W, M, Q, h, and B all have consistent self-comparison:
   - D, W, h, B: both return `True`
   - M, Q: both return `False`
   - Only Y has this inconsistency

4. **User Confusion**: Users relying on the mathematical relationship between these functions will get incorrect results when working with yearly frequencies.

## Fix

The fix depends on the intended semantics. Looking at the pattern for other frequencies, it appears the issue is in one of the implementations. The fix should make both functions return the same value for self-comparison of 'Y'.

Without seeing the implementation, the most likely fix is to ensure both functions use the same logic for determining whether a frequency is a sub/superperiod of itself.