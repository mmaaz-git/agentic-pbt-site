# Bug Report: pandas.tseries.frequencies Symmetry Violation in is_subperiod/is_superperiod

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(source, target)` and `is_superperiod(target, source)` violate the fundamental symmetry property that should hold between downsampling and upsampling. Specifically, for frequency pairs involving daily ('D') and business day ('B') frequencies, `is_subperiod('D', 'B')` returns `False` while `is_superperiod('B', 'D')` returns `True`, breaking the expected symmetry.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQ_STRINGS = [
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'ME', 'QE', 'YE',
    'B', 'BME', 'BQE', 'BYE',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
    'MS', 'QS', 'YS', 'BMS', 'BQS', 'BYS',
]

@given(
    source=st.sampled_from(VALID_FREQ_STRINGS),
    target=st.sampled_from(VALID_FREQ_STRINGS)
)
@settings(max_examples=500)
def test_subperiod_superperiod_symmetry(source, target):
    try:
        is_sub = freq.is_subperiod(source, target)
        is_super = freq.is_superperiod(target, source)

        assert is_sub == is_super, (
            f"Symmetry broken: is_subperiod({source}, {target}) = {is_sub}, "
            f"but is_superperiod({target}, {source}) = {is_super}"
        )
    except (ValueError, KeyError) as e:
        pass
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

is_sub = freq.is_subperiod('D', 'B')
is_super = freq.is_superperiod('B', 'D')

print(f"is_subperiod('D', 'B') = {is_sub}")
print(f"is_superperiod('B', 'D') = {is_super}")
print(f"Symmetry violated: {is_sub} != {is_super}")
```

Output:
```
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
Symmetry violated: False != True
```

## Why This Is A Bug

According to the docstrings:
- `is_subperiod(source, target)` returns `True` if downsampling is possible between source and target frequencies
- `is_superperiod(source, target)` returns `True` if upsampling is possible between source and target frequencies

By definition, if downsampling from frequency A to frequency B is possible, then upsampling from frequency B to frequency A must also be possible. These are inverse operations. Therefore, the symmetry property `is_subperiod(A, B) == is_superperiod(B, A)` should always hold.

For the specific case of 'D' (daily) and 'B' (business day):
- Daily frequency includes all days: Mon, Tue, Wed, Thu, Fri, Sat, Sun
- Business day frequency includes only weekdays: Mon, Tue, Wed, Thu, Fri
- You can downsample from 'D' to 'B' by filtering out weekends
- You can upsample from 'B' to 'D' by adding weekend days

Both operations are possible, so both `is_subperiod('D', 'B')` and `is_superperiod('B', 'D')` should return the same value (either both `True` or both `False`). The current implementation returns different values, violating this fundamental symmetry.

## Fix

This bug likely stems from incomplete handling of business day frequencies in either `is_subperiod` or `is_superperiod`. The fix would require examining the implementation of both functions to ensure they properly recognize the relationship between daily and business day frequencies. Specifically, `is_subperiod('D', 'B')` should return `True` to match the correct behavior of `is_superperiod('B', 'D')`, or both should return `False` if pandas considers these frequencies incomparable for resampling purposes.