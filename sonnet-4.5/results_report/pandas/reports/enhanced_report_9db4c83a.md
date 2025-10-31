# Bug Report: pandas.tseries.api.guess_datetime_format Incorrectly Swaps Day and Month for ISO Dates with dayfirst=True

**Target**: `pandas.tseries.api.guess_datetime_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `guess_datetime_format` function incorrectly guesses `%Y-%d-%m` instead of `%Y-%m-%d` for ISO-formatted dates when `dayfirst=True`, causing dates to be parsed with day and month swapped, resulting in silent data corruption.

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

if __name__ == "__main__":
    test_guess_datetime_format_roundtrip()
```

<details>

<summary>
**Failing input**: `dt=datetime.datetime(2000, 1, 2, 0, 0), fmt='%Y-%m-%d', dayfirst=True`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/27/hypo.py:20: UserWarning: Parsing dates in %d/%m/%Y format when dayfirst=False (the default) was specified. Pass `dayfirst=True` or specify a format to silence this warning.
  guessed_fmt = guess_datetime_format(dt_str, dayfirst=dayfirst)
/home/npc/pbt/agentic-pbt/worker_/27/hypo.py:20: UserWarning: Parsing dates in %Y-%m-%d format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
  guessed_fmt = guess_datetime_format(dt_str, dayfirst=dayfirst)
/home/npc/pbt/agentic-pbt/worker_/27/hypo.py:20: UserWarning: Parsing dates in %Y/%m/%d format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
  guessed_fmt = guess_datetime_format(dt_str, dayfirst=dayfirst)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 28, in <module>
    test_guess_datetime_format_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 13, in test_guess_datetime_format_roundtrip
    st.datetimes(min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 24, in test_guess_datetime_format_roundtrip
    assert parsed.date() == dt.date(), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: datetime.strptime('2000-01-02', '%Y-%d-%m').date() = 2000-02-01 != 2000-01-02
Falsifying example: test_guess_datetime_format_roundtrip(
    dt=datetime.datetime(2000, 1, 2, 0, 0),
    fmt='%Y-%m-%d',
    dayfirst=True,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/27/hypo.py:25
```
</details>

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

<details>

<summary>
AssertionError: Parsed date 2000-02-01 does not match expected 2000-01-02
</summary>
```
Input: 2000-01-02
Guessed format: %Y-%d-%m
Parsed date: 2000-02-01
Expected date: 2000-01-02
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/repo.py", line 14, in <module>
    assert parsed.date() == datetime(2000, 1, 2).date()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```
</details>

## Why This Is A Bug

This violates the fundamental contract of `guess_datetime_format`: the returned format string should correctly parse the input string back to its original date value. The bug causes silent data corruption by swapping day and month values, turning January 2nd into February 1st.

The documentation acknowledges this as a "known bug" with the warning: "dayfirst=True is not strict, but will prefer to parse with day first (this is a known bug)." However, this doesn't justify returning a format that produces incorrect results.

Key problems:
1. **ISO 8601 dates are unambiguous**: When a 4-digit year appears first (YYYY-MM-DD), the format is internationally standardized and should never have day/month swapped
2. **Silent data corruption**: The function returns a valid format string that parses without error but produces wrong dates
3. **Inconsistent behavior**: The function correctly handles unambiguous dates (where day > 12) but fails for ambiguous ones (day <= 12)
4. **Affects core pandas functions**: This bug propagates to `pandas.to_datetime()` when used with `dayfirst=True`

## Relevant Context

The bug demonstrates inconsistent logic in the parsing algorithm:

```python
# When day > 12 (unambiguous), it works correctly:
guess_datetime_format("2000-12-25", dayfirst=True)  # Returns '%Y-%m-%d' (correct)
guess_datetime_format("2000-25-12", dayfirst=True)  # Returns '%Y-%d-%m' (correct)

# When day <= 12 (ambiguous), it incorrectly applies dayfirst to ISO dates:
guess_datetime_format("2000-01-02", dayfirst=True)  # Returns '%Y-%d-%m' (incorrect)
guess_datetime_format("2000-01-02", dayfirst=False) # Returns '%Y-%m-%d' (correct)
```

This also affects `pandas.to_datetime()`:
```python
import pandas as pd
pd.to_datetime(['2000-01-02'], dayfirst=True)  # Returns 2000-02-01 (incorrect)
```

Documentation references:
- Function source: `pandas._libs.tslibs.parsing` (Cython implementation)
- pandas.to_datetime documentation: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
- ISO 8601 standard: https://en.wikipedia.org/wiki/ISO_8601

## Proposed Fix

The fix requires modifying the Cython implementation to recognize year-first formats and not apply `dayfirst` logic to them. Since the exact implementation is in Cython, here's a high-level approach:

1. Detect if the date format starts with a 4-digit year (YYYY)
2. If yes, treat it as an ISO-style date and ignore the `dayfirst` parameter for determining field order
3. Only apply `dayfirst` logic to ambiguous formats like DD/MM/YYYY vs MM/DD/YYYY

The logic should be:
- If format matches `YYYY-*-*` or `YYYY/*/*` pattern, always interpret as Year-Month-Day regardless of `dayfirst`
- For other formats, apply existing `dayfirst` logic

This would require changes in the `_guess_datetime_format` function in `pandas/_libs/tslibs/parsing.pyx` to add a check for year-first formats before applying the dayfirst swap logic.