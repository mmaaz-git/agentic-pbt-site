# Bug Report: pandas.plotting.scatter_matrix crashes with single-row DataFrames containing large values

**Target**: `pandas.plotting.scatter_matrix`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scatter_matrix` function crashes with a `ValueError` when plotting single-row DataFrames containing large numeric values (magnitude >= 1e15), due to matplotlib's histogram failing to create finite-sized bins for values with extreme floating-point precision issues.

## Property-Based Test

```python
from hypothesis import given, assume, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@settings(max_examples=200, deadline=None)
@given(
    df=data_frames(
        columns=[
            column('A', dtype=float),
            column('B', dtype=float),
            column('C', dtype=float),
        ],
        index=range_indexes(min_size=1, max_size=50)
    )
)
def test_scatter_matrix_shape_property(df):
    assume(not df.empty)
    assume(not df.isna().all().all())

    result = pd.plotting.scatter_matrix(df)
    n_cols = len(df.columns)
    assert result.shape == (n_cols, n_cols)
    plt.close('all')

# Run the test
if __name__ == "__main__":
    test_scatter_matrix_shape_property()
```

<details>

<summary>
**Failing input**: `DataFrame({'A': {0: 0.0}, 'B': {0: -1.297501e+16}, 'C': {0: -1.297501e+16}})`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:91: UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
  ax.set_xlim(boundaries_list[i])
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:100: UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
  ax.set_xlim(boundaries_list[j])
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:101: UserWarning: Attempting to set identical low and high ylims makes transformation singular; automatically expanding.
  ax.set_ylim(boundaries_list[i])
Traceback (most recent call last):
  File "<string>", line 27, in test_scatter_matrix_shape_property
    result = pd.plotting.scatter_matrix(df)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py", line 220, in scatter_matrix
    return plot_backend.scatter_matrix(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        frame=frame,
        ^^^^^^^^^^^^
    ...<9 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py", line 81, in scatter_matrix
    ax.hist(values, **hist_kwds)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/_api/deprecation.py", line 453, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/__init__.py", line 1521, in inner
    return func(
        ax,
        *map(cbook.sanitize_sequence, args),
        **{k: cbook.sanitize_sequence(v) for k, v in kwargs.items()})
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/axes/_axes.py", line 7129, in hist
    m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
              ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_histograms_impl.py", line 792, in histogram
    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
                              ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_histograms_impl.py", line 449, in _get_bin_edges
    raise ValueError(
        f'Too many bins for data range. Cannot create {n_equal_bins} '
        f'finite-sized bins.')
ValueError: Too many bins for data range. Cannot create 10 finite-sized bins.
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a single-row DataFrame with specific numeric values
df = pd.DataFrame({
    'A': [0.0],
    'B': [-1.297501e+16],
    'C': [-1.297501e+16]
})

# This should crash with ValueError
pd.plotting.scatter_matrix(df)
```

<details>

<summary>
ValueError: Too many bins for data range
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:91: UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
  ax.set_xlim(boundaries_list[i])
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:100: UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
  ax.set_xlim(boundaries_list[j])
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:101: UserWarning: Attempting to set identical low and high ylims makes transformation singular; automatically expanding.
  ax.set_ylim(boundaries_list[i])
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/repo.py", line 14, in <module>
    pd.plotting.scatter_matrix(df)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py", line 220, in scatter_matrix
    return plot_backend.scatter_matrix(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        frame=frame,
        ^^^^^^^^^^^^
    ...<9 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py", line 81, in scatter_matrix
    ax.hist(values, **hist_kwds)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/_api/deprecation.py", line 453, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/__init__.py", line 1521, in inner
    return func(
        ax,
        *map(cbook.sanitize_sequence, args),
        **{k: cbook.sanitize_sequence(v) for k, v in kwargs.items()})
  File "/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/axes/_axes.py", line 7129, in hist
    m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
              ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_histograms_impl.py", line 792, in histogram
    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
                              ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_histograms_impl.py", line 449, in _get_bin_edges
    raise ValueError(
        f'Too many bins for data range. Cannot create {n_equal_bins} '
        f'finite-sized bins.')
ValueError: Too many bins for data range. Cannot create 10 finite-sized bins.
```
</details>

## Why This Is A Bug

1. **Valid input crashes**: Single-row DataFrames are valid pandas DataFrame objects according to the type signature and documentation. The function accepts them as input but crashes unpredictably based on the numeric values contained.

2. **Inconsistent behavior**: The function successfully processes some single-row DataFrames (with smaller values) but crashes with others (values >= 1e15), creating an inconsistency that violates the principle of least surprise.

3. **Undocumented limitation**: Neither the pandas documentation for `scatter_matrix` nor the underlying matplotlib histogram documentation specifies this limitation on single-row DataFrames or extreme numeric values.

4. **Floating-point precision issue**: The crash occurs when matplotlib's histogram function encounters floating-point precision problems while trying to create 10 finite-sized bins for a single unique value at extreme magnitudes. With large values like 1e16, the floating-point representation cannot distinguish between adjacent bin edges.

5. **Silent contract violation**: The function signature and docstring implicitly promise to handle any numeric DataFrame, but this edge case violates that contract without proper error handling or documentation.

## Relevant Context

The issue originates in the histogram creation on the diagonal of the scatter matrix at line 81 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py`:
```python
ax.hist(values, **hist_kwds)
```

When all values in a column are identical (as in single-row DataFrames), and these values are extremely large (>= 1e15), matplotlib's `np.histogram` function cannot create finite-sized bins due to floating-point arithmetic limitations. The data range is effectively zero, but at extreme scales, floating-point imprecision prevents proper bin edge calculation.

The function also generates multiple warnings about "Attempting to set identical low and high xlims" before the crash, indicating problems with axis limit calculations for single-value datasets.

Related pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html

## Proposed Fix

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -77,8 +77,17 @@ def scatter_matrix(
                 values = df[a].values[mask[a].values]

                 # Deal with the diagonal by drawing a histogram there.
                 if diagonal == "hist":
-                    ax.hist(values, **hist_kwds)
+                    # Handle edge case where histogram fails for single values
+                    # or extreme values with floating-point precision issues
+                    try:
+                        ax.hist(values, **hist_kwds)
+                    except ValueError as e:
+                        if "Too many bins" in str(e) or len(values) == 1:
+                            # Fall back to a simple vertical line for single values
+                            ax.axvline(values[0], color='blue', linewidth=2)
+                            ax.set_ylabel('Count')
+                        else:
+                            raise

                 elif diagonal in ("kde", "density"):
                     from scipy.stats import gaussian_kde
```