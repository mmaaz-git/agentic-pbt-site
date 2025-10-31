# Bug Report: pandas.plotting.scatter_matrix diagonal parameter validation

**Target**: `pandas.plotting.scatter_matrix`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scatter_matrix` function accepts invalid values for the `diagonal` parameter, violating its documented API contract that explicitly states only 'hist' or 'kde' are valid values.

## Property-Based Test

```python
@settings(max_examples=200)
@given(
    df=data_frames(
        columns=columns(['a', 'b', 'c'], dtype=float),
        index=range_indexes(min_size=2, max_size=20)
    ),
    diagonal=st.text(min_size=1, max_size=10).filter(lambda x: x not in ['hist', 'kde'])
)
def test_scatter_matrix_diagonal_validation(df, diagonal):
    """
    Property: scatter_matrix should only accept 'hist' or 'kde' for diagonal parameter.
    The docstring explicitly states: diagonal : {'hist', 'kde'}
    """
    assume(len(df) >= 2)
    assume(len(df.columns) >= 2)

    try:
        result = pandas.plotting.scatter_matrix(df, diagonal=diagonal)
        assert False, f"scatter_matrix should reject diagonal='{diagonal}', but it didn't"
    except (ValueError, KeyError) as e:
        pass
```

**Failing input**: `diagonal='0'` (or any other string besides 'hist' or 'kde')

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'a': [1.0, 2.0],
    'b': [3.0, 4.0],
    'c': [5.0, 6.0]
})

result = pandas.plotting.scatter_matrix(df, diagonal='invalid')
print(f"Accepted invalid diagonal='invalid'")

result = pandas.plotting.scatter_matrix(df, diagonal='0')
print(f"Accepted invalid diagonal='0'")

result = pandas.plotting.scatter_matrix(df, diagonal='foobar')
print(f"Accepted invalid diagonal='foobar'")
```

## Why This Is A Bug

The function's docstring explicitly states:

```
diagonal : {'hist', 'kde'}
    Pick between 'kde' and 'hist' for either Kernel Density Estimation or
    Histogram plot in the diagonal.
```

The notation `{'hist', 'kde'}` is Python's standard way of documenting that a parameter must be one of a specific set of values. However, the function accepts any string value without validation. This violates the API contract and can lead to:

1. Confusing behavior when users pass typos or incorrect values
2. Silent failures where the plot is generated incorrectly without errors
3. Users not discovering their mistakes until they examine the output

## Fix

Add validation at the beginning of the `scatter_matrix` function:

```diff
def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Mapping[str, Any] | None = None,
    hist_kwds: Mapping[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwargs,
) -> np.ndarray:
+    if diagonal not in ('hist', 'kde'):
+        raise ValueError(
+            f"diagonal must be 'hist' or 'kde', got {diagonal!r}"
+        )
+
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.scatter_matrix(
        frame=frame,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        grid=grid,
        diagonal=diagonal,
        marker=marker,
        density_kwds=density_kwds,
        hist_kwds=hist_kwds,
        range_padding=range_padding,
        **kwargs,
    )
```