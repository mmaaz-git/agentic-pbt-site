# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation for Day/BusinessDay

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate the inverse relationship property for Day ('D') and BusinessDay ('B') frequencies. Specifically, `is_subperiod('D', 'B')` returns `False` while `is_superperiod('B', 'D')` returns `True`. By definition, if A is a subperiod of B, then B must be a superperiod of A.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from(['D', 'W', 'M', 'Q', 'Y', 'h', 'B'])

@given(freq_strings, freq_strings)
@settings(max_examples=100)
def test_is_subperiod_inverse_of_superperiod(freq1, freq2):
    sub_result = is_subperiod(freq1, freq2)
    super_result = is_superperiod(freq2, freq1)

    assert sub_result == super_result, \
        f"Not inverse: is_subperiod({freq1}, {freq2})={sub_result} != is_superperiod({freq2}, {freq1})={super_result}"
```

**Failing input**: `freq1='D'`, `freq2='B'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

sub = is_subperiod('D', 'B')
sup = is_superperiod('B', 'D')

print(f"is_subperiod('D', 'B') = {sub}")
print(f"is_superperiod('B', 'D') = {sup}")
assert sub == sup
```

Output:
```
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
AssertionError
```

## Why This Is A Bug

This violates the fundamental definition of inverse relationships:

1. **Mathematical Definition**: By definition, A is a subperiod of B if and only if B is a superperiod of A. These are inverse relationships, so:
   - `is_subperiod(A, B) == is_superperiod(B, A)` must always hold
   - Violating this breaks the mathematical foundation of the API

2. **Logical Inconsistency**: The current behavior states:
   - Day is NOT a subperiod of BusinessDay
   - BusinessDay IS a superperiod of Day
   - This is logically contradictory

3. **Other frequency pairs work correctly**: Testing various combinations shows that most pairs respect the inverse relationship:
   - (D, W), (D, M), (D, Q), (D, Y), (M, Q), (Q, Y), (h, D) all work correctly
   - Only (D, B) exhibits this bug

4. **User Impact**: Users cannot rely on the documented mathematical relationship between these functions. Code that uses the property `is_subperiod(A, B) ⟺ is_superperiod(B, A)` will produce incorrect results.

5. **Semantic Question**: There's an underlying question about whether:
   - BusinessDay should be considered a subperiod of Day (every business day is a day), OR
   - Day should be considered a subperiod of BusinessDay (not all days are business days)

   Either interpretation is valid, but both functions must agree.

## Fix

The fix requires deciding on the correct semantic relationship between Day and BusinessDay, then ensuring both functions implement it consistently.

The most logical interpretation is that BusinessDay is a subperiod (subset) of Day, since business days are a subset of all days:
- Every business day is a day (true)
- Not every day is a business day (false)

This suggests:
- `is_subperiod('B', 'D')` should return `True`
- `is_superperiod('D', 'B')` should return `True`
- `is_subperiod('D', 'B')` should return `False`
- `is_superperiod('B', 'D')` should return `False`

The current bug is that `is_superperiod('B', 'D')` returns `True` when it should return `False` (or alternatively, `is_subperiod('D', 'B')` should return `True`).