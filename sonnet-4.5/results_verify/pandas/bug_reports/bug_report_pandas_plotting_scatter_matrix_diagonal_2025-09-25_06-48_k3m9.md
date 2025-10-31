# Bug Report: pandas.plotting.scatter_matrix Diagonal Parameter Validation

**Target**: `pandas.plotting.scatter_matrix`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scatter_matrix` function silently accepts invalid values for the `diagonal` parameter, resulting in empty diagonal plots instead of raising a validation error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@given(
    df=hpd.data_frames(
        columns=[
            hpd.column('A', elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
            hpd.column('B', elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        ],
        index=hpd.range_indexes(min_size=2, max_size=50)
    ),
    invalid_diagonal=st.text(min_size=1, max_size=10).filter(lambda x: x not in ['hist', 'kde', 'density'])
)
@settings(max_examples=50, deadline=None)
def test_scatter_matrix_invalid_diagonal(df, invalid_diagonal):
    try:
        result = pd.plotting.scatter_matrix(df, diagonal=invalid_diagonal)
        plt.close('all')
    except (ValueError, KeyError) as e:
        plt.close('all')
        return

    raise AssertionError(f"Expected error for invalid diagonal '{invalid_diagonal}', but got success")
```

**Failing input**: `pd.plotting.scatter_matrix(df, diagonal='invalid')`

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 4.0, 5.0],
    'B': [2.0, 3.0, 4.0, 5.0, 6.0],
    'C': [3.0, 4.0, 5.0, 6.0, 7.0]
})

result = pd.plotting.scatter_matrix(df, diagonal='invalid_value')

fig = plt.gcf()
axes = fig.get_axes()
diagonal_axes = [axes[i*3 + i] for i in range(3)]
for i, ax in enumerate(diagonal_axes):
    children = ax.get_children()
    print(f"Diagonal axis {i} has {len(children)} elements (should have more if diagonal plot was drawn)")

plt.close('all')
```

## Why This Is A Bug

The function's docstring explicitly states that the `diagonal` parameter accepts `{'hist', 'kde'}`, but the implementation silently accepts any value:

1. **Documentation says**: `diagonal : {'hist', 'kde'} Pick between 'kde' and 'hist' for either Kernel Density Estimation or Histogram plot in the diagonal.`

2. **Actual behavior**: Any value is accepted, but invalid values result in empty diagonal plots with no error or warning.

This violates the API contract in two ways:
- No validation of the documented allowed values
- Silent failure instead of clear error messaging

Users who make typos (e.g., `diagonal='histogram'` or `diagonal='density'`) will get silently broken output instead of a helpful error message.

## Fix

```diff
def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax=None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds=None,
    hist_kwds=None,
    range_padding: float = 0.05,
    **kwds,
):
+   valid_diagonals = {"hist", "kde", "density"}
+   if diagonal not in valid_diagonals:
+       raise ValueError(
+           f"diagonal must be one of {valid_diagonals}, got {diagonal!r}"
+       )
+
    df = frame._get_numeric_data()
    n = df.columns.size
    # ... rest of function
```