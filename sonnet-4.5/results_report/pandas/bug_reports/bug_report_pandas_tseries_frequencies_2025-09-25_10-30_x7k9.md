# Bug Report: pandas.tseries.frequencies Inconsistent Frequency Relationship

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod()` and `is_superperiod()` return logically inconsistent results for the frequency pair 'D' (day) and 'B' (business day). Both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')` return True, which is mathematically impossible.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries import frequencies

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'ME', 'MS', 'QE', 'QS', 'YE', 'YS',
    'B', 'BME', 'BMS', 'BQE', 'BQS', 'BYE', 'BYS'
])

@given(freq_strings, freq_strings)
@settings(max_examples=500)
def test_subperiod_superperiod_inverse_relationship(source, target):
    is_sub = frequencies.is_subperiod(source, target)
    is_super = frequencies.is_superperiod(target, source)
    assert is_sub == is_super
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
from pandas.tseries import frequencies

is_sub_d_b = frequencies.is_subperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')

assert is_sub_d_b == is_super_b_d

is_super_d_b = frequencies.is_superperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')

assert not (is_super_d_b and is_super_b_d)
```

## Why This Is A Bug

The inverse relationship between `is_subperiod` and `is_superperiod` is violated:
- `is_subperiod('D', 'B')` returns `False`, meaning we cannot downsample from day to business day
- `is_superperiod('B', 'D')` returns `True`, meaning we can upsample from business day to day

These should be equivalent (both True or both False) because they describe the same frequency relationship from opposite directions.

Additionally, both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')` return `True`, which is logically impossible. Two frequencies cannot simultaneously be superperiods of each other, as this would imply each is both more and less frequent than the other.

## Fix

The issue likely stems from how business day frequency is handled in the comparison logic. Since 'D' (calendar day) is more frequent than 'B' (business day - only 5 days per week), the correct behavior should be:

- `is_subperiod('D', 'B')` should return `True` (can downsample daily to business daily)
- `is_superperiod('B', 'D')` should return `True` (can upsample business daily to daily)
- `is_superperiod('D', 'B')` should return `False`
- `is_subperiod('B', 'D')` should return `False`

The fix requires examining the frequency comparison logic in the implementation of these functions to ensure proper handling of the 'B' (business day) frequency in relation to 'D' (calendar day).