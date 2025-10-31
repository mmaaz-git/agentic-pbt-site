# Bug Report: pandas.tseries.api.guess_datetime_format dayfirst Parameter Incorrectly Guesses Format

**Target**: `pandas.tseries.api.guess_datetime_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `guess_datetime_format` function incorrectly guesses the datetime format when `dayfirst=True` is specified. For the input string "2000-01-02" with `dayfirst=True`, it returns `'%Y-%d-%m'` instead of `'%Y-%m-%d'`, causing the date to be parsed incorrectly as February 1st instead of January 2nd.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.api import guess_datetime_format
from datetime import datetime

DATETIME_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
]

@given(
    st.datetimes(min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31)),
    st.sampled_from(DATETIME_FORMATS),
    st.booleans()
)
@settings(max_examples=500)
def test_guess_datetime_format_roundtrip(dt, fmt, dayfirst):
    dt_str = dt.strftime(fmt)
    guessed_fmt = guess_datetime_format(dt_str, dayfirst=dayfirst)

    if guessed_fmt is not None:
        parsed = datetime.strptime(dt_str, guessed_fmt)
        assert parsed.date() == dt.date(), \
            f"Round-trip failed: datetime.strptime({dt_str!r}, {guessed_fmt!r}).date() = {parsed.date()} != {dt.date()}"
```

**Failing input**: `dt=datetime(2000, 1, 2, 0, 0), fmt='%Y-%m-%d', dayfirst=True`

## Reproducing the Bug

```python
from pandas.tseries.api import guess_datetime_format
from datetime import datetime

dt_str = "2000-01-02"
guessed_fmt = guess_datetime_format(dt_str, dayfirst=True)

print(f"Input: {dt_str}")
print(f"Guessed format: {guessed_fmt}")

parsed = datetime.strptime(dt_str, guessed_fmt)
print(f"Parsed date: {parsed.date()}")
print(f"Expected date: 2000-01-02")

assert parsed.date() == datetime(2000, 1, 2).date()
```

**Output**:
```
Input: 2000-01-02
Guessed format: %Y-%d-%m
Parsed date: 2000-02-01
Expected date: 2000-01-02
AssertionError
```

## Why This Is A Bug

The function's purpose is to guess the correct datetime format string that can be used with `strptime` to parse the input. When the guessed format fails to correctly parse the input string back to the original date, the function has failed its core contract.

The docstring warns that "dayfirst=True is not strict", but returning a format that produces incorrect parsing results is a bug, not just non-strict behavior. The format `%Y-%d-%m` interprets "2000-01-02" as "year 2000, day 01, month 02" (February 1st) instead of the intended "year 2000, month 01, day 02" (January 2nd).

## Fix

The issue requires investigating the implementation of `guess_datetime_format` to understand why it's guessing `%Y-%d-%m` when `dayfirst=True`. The function should either:

1. Correctly guess `%Y-%m-%d` (since the year-first format is unambiguous), or
2. Return `None` if it cannot determine the correct format

A proper fix would require examining the Cython implementation to understand the guessing logic and ensure it respects the actual date separator positions rather than just the `dayfirst` flag when the format is clearly year-first.