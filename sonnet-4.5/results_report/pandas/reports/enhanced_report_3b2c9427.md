# Bug Report: pandas.plotting.autocorrelation_plot Crashes with ZeroDivisionError on Empty Series

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot()` function crashes with an uninformative `ZeroDivisionError` when called with an empty Series, instead of raising a meaningful error message that helps users understand the actual issue.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis that discovered the bug in
pandas.plotting.autocorrelation_plot when given empty Series
"""

from hypothesis import given, strategies as st, example
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=100))
@example([])  # Explicitly test the empty case
def test_autocorrelation_plot_handles_empty(data):
    """
    Test that autocorrelation_plot handles edge cases gracefully,
    especially empty Series which should either work or raise a meaningful error
    """
    series = pd.Series(data)
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.autocorrelation_plot(series)
        # If it succeeds, the result should be valid
        assert result is not None
        print(f"✓ Success with {len(data)} elements")
    except ValueError as e:
        # If it fails, it should raise a meaningful ValueError
        assert "empty" in str(e).lower() or "length" in str(e).lower()
        print(f"✓ Raised meaningful ValueError for {len(data)} elements: {e}")
    except ZeroDivisionError as e:
        # This should NOT happen - it's the bug we're reporting
        print(f"✗ BUG: ZeroDivisionError with {len(data)} elements: {e}")
        raise AssertionError(f"Function crashed with ZeroDivisionError instead of handling empty series gracefully: {e}")
    except Exception as e:
        print(f"✗ Unexpected error with {len(data)} elements: {type(e).__name__}: {e}")
        raise
    finally:
        plt.close(fig)

if __name__ == "__main__":
    # Run the test
    test_autocorrelation_plot_handles_empty()
```

<details>

<summary>
**Failing input**: `[]`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3859: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py:146: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret / rcount
✗ BUG: ZeroDivisionError with 0 elements: division by zero
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 22, in test_autocorrelation_plot_handles_empty
    result = pandas.plotting.autocorrelation_plot(series)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py", line 605, in autocorrelation_plot
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py", line 454, in autocorrelation_plot
    c0 = np.sum((data - mean) ** 2) / n
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 42, in <module>
    test_autocorrelation_plot_handles_empty()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 13, in test_autocorrelation_plot_handles_empty
    @example([])  # Explicitly test the empty case
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 33, in test_autocorrelation_plot_handles_empty
    raise AssertionError(f"Function crashed with ZeroDivisionError instead of handling empty series gracefully: {e}")
AssertionError: Function crashed with ZeroDivisionError instead of handling empty series gracefully: division by zero
Falsifying explicit example: test_autocorrelation_plot_handles_empty(
    data=[],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the pandas.plotting.autocorrelation_plot bug
with empty Series causing ZeroDivisionError
"""

import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

# Create an empty series
empty_series = pd.Series([])

# Create figure for plotting
fig, ax = plt.subplots()

try:
    # This should either work gracefully or raise a meaningful error
    # Instead it crashes with ZeroDivisionError
    result = pandas.plotting.autocorrelation_plot(empty_series)
    print("Success: Function returned:", result)
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error ({type(e).__name__}): {e}")
finally:
    plt.close(fig)
```

<details>

<summary>
ZeroDivisionError: division by zero
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3859: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py:146: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret / rcount
ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior because the function fails to validate its input before performing mathematical operations that require non-empty data. The documentation for `autocorrelation_plot` does not specify any minimum size requirements or warn about potential ZeroDivisionError with empty Series. Users reasonably expect either:

1. **Proper input validation** with a clear, descriptive ValueError explaining that autocorrelation requires at least one data point
2. **Documentation** that explicitly states empty Series are not supported

Instead, the function crashes with a confusing ZeroDivisionError that doesn't indicate the actual problem (empty input). The error occurs at line 454 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py`:

```python
c0 = np.sum((data - mean) ** 2) / n  # n is 0 for empty series
```

Additionally, NumPy warnings about "Mean of empty slice" and "invalid value encountered in scalar divide" appear before the crash, further confusing the error output. Autocorrelation mathematically requires data points to calculate correlations between a series and its lagged values, making empty input inherently invalid for this operation.

## Relevant Context

- **Code location**: `/pandas/plotting/_matplotlib/misc.py`, function `autocorrelation_plot`, lines 444-474
- **Direct cause**: Line 454 divides by `n` (series length) without checking if `n > 0`
- **Mathematical context**: Autocorrelation measures how a time series correlates with itself at different lags, requiring:
  - At least one data point to calculate mean
  - At least one data point to calculate variance (c0)
  - Multiple data points for meaningful lag correlations
- **Similar pandas functions**: Most pandas functions that require non-empty input provide clear ValueError messages
- **Documentation gap**: Function docstring doesn't mention minimum size requirements or potential errors

## Proposed Fix

```diff
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    import matplotlib.pyplot as plt

    n = len(series)
+   if n == 0:
+       raise ValueError("autocorrelation_plot requires a non-empty Series")
+
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n
```