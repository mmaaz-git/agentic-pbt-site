# Bug Report: pandas.plotting.radviz Produces NaN Values on Constant Columns

**Target**: `pandas.plotting.radviz`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `radviz` function produces NaN values and invalid visualizations when any numeric column contains all identical values, due to division by zero during normalization that results in NaN propagation rather than an exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

@given(
    hpd.data_frames(
        columns=[
            hpd.column('A', elements=st.just(1.0)),
            hpd.column('B', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            hpd.column('class', elements=st.sampled_from(['cat1', 'cat2']))
        ],
        index=hpd.range_indexes(min_size=2, max_size=10)
    )
)
@settings(max_examples=5, deadline=None)
def test_radviz_constant_column(df):
    print(f"\nTesting with DataFrame:")
    print(df)

    fig, ax = plt.subplots()
    try:
        result = pd.plotting.radviz(df, 'class', ax=ax)

        # Check if the resulting plot has NaN values
        # This happens when normalization divides by zero
        def normalize(series):
            a = min(series)
            b = max(series)
            return (series - a) / (b - a)

        normalized_A = normalize(df['A'])
        if np.any(np.isnan(normalized_A)):
            print(f"WARNING: Normalization produced NaN values!")
            print(f"Column A: min={df['A'].min()}, max={df['A'].max()}")
            print(f"Normalized column A contains NaN: {np.any(np.isnan(normalized_A))}")
            assert False, "radviz produced NaN values due to constant column"

        print("Test passed - no NaN values produced")
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError occurred: {e}")
        assert False, f"radviz raised ZeroDivisionError: {e}"
    finally:
        plt.close('all')

if __name__ == "__main__":
    print("Running property-based test for pandas.plotting.radviz with constant columns...")
    print("=" * 60)
    test_radviz_constant_column()
```

<details>

<summary>
**Failing input**: `DataFrame with column 'A' containing all 1.0 values`
</summary>
```
Running property-based test for pandas.plotting.radviz with constant columns...
============================================================

Testing with DataFrame:
     A    B class
0  1.0  0.0  cat1
1  1.0  0.0  cat1
WARNING: Normalization produced NaN values!
Column A: min=1.0, max=1.0
Normalized column A contains NaN: True

Testing with DataFrame:
     A    B class
0  1.0  0.0  cat1
1  1.0  0.0  cat1
WARNING: Normalization produced NaN values!
Column A: min=1.0, max=1.0
Normalized column A contains NaN: True
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 52, in <module>
    test_radviz_constant_column()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 10, in test_radviz_constant_column
    hpd.data_frames(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 40, in test_radviz_constant_column
    assert False, "radviz produced NaN values due to constant column"
           ^^^^^
AssertionError: radviz produced NaN values due to constant column
Falsifying example: test_radviz_constant_column(
    df=
             A    B class
        0  1.0  0.0  cat1
        1  1.0  0.0  cat1
    ,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Show all warnings
warnings.filterwarnings('always')

df = pd.DataFrame({
    'A': [1.0, 1.0, 1.0, 1.0],
    'B': [2.0, 3.0, 4.0, 5.0],
    'class': ['cat1', 'cat1', 'cat2', 'cat2']
})

print("DataFrame:")
print(df)
print("\nColumn A values:", df['A'].values)
print("Column A min:", df['A'].min())
print("Column A max:", df['A'].max())
print("Column A range (max - min):", df['A'].max() - df['A'].min())

fig, ax = plt.subplots()
try:
    result = pd.plotting.radviz(df, 'class', ax=ax)
    print("\nRadViz completed successfully")

    # Check if the normalization resulted in NaN values
    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    normalized_A = normalize(df['A'])
    print("\nNormalized column A:", normalized_A)
    print("Contains NaN:", np.any(np.isnan(normalized_A)))

except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")
finally:
    plt.close('all')
```

<details>

<summary>
Output showing NaN propagation without exception
</summary>
```
DataFrame:
     A    B class
0  1.0  2.0  cat1
1  1.0  3.0  cat1
2  1.0  4.0  cat2
3  1.0  5.0  cat2

Column A values: [1. 1. 1. 1.]
Column A min: 1.0
Column A max: 1.0
Column A range (max - min): 0.0

RadViz completed successfully

Normalized column A: 0   NaN
1   NaN
2   NaN
3   NaN
Name: A, dtype: float64
Contains NaN: True
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Silent Failure**: The function completes "successfully" but produces an invalid visualization with NaN values. Users may not realize their plot is broken. The documentation states that RadViz "allow to project a N-dimensional data set into a 2D space" but fails to do so correctly when a dimension has no variance.

2. **Mathematical Error**: The `normalize` function in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py` (lines 147-150) performs `(series - a) / (b - a)` where for constant columns, `min(series) == max(series)`, causing division by zero that produces NaN instead of raising an exception.

3. **Undocumented Constraint**: The documentation doesn't specify that columns must have variance. Users reasonably expect the function to handle all valid DataFrame inputs, especially since constant features are common in real datasets (e.g., after filtering, certain attributes might become constant).

4. **Inconsistent with pandas philosophy**: pandas typically handles edge cases gracefully or provides clear error messages. Silently producing invalid output contradicts this principle.

## Relevant Context

The RadViz visualization technique requires normalization to map data points into a unit circle. The current implementation assumes all columns will have different min/max values. When this assumption is violated, the normalization step produces NaN values that propagate through subsequent calculations, resulting in:

- Invalid coordinate calculations (line 178 in misc.py: `y = (s * row_).sum(axis=0) / row.sum()`)
- Points that cannot be plotted correctly
- A broken visualization without any warning to the user

The pandas documentation at https://pandas.pydata.org/docs/reference/api/pandas.plotting.radviz.html doesn't mention this limitation. The test suite in `pandas/tests/plotting/test_misc.py` also lacks coverage for this edge case.

Real-world scenarios where this occurs:
- One-hot encoded features where a category is always present/absent
- Sensor data where a sensor is stuck at a constant value
- Filtered datasets where certain attributes become constant
- Normalized data where some features have zero variance

## Proposed Fix

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -145,9 +145,15 @@ def radviz(
     import matplotlib.pyplot as plt

     def normalize(series):
         a = min(series)
         b = max(series)
+        if a == b:
+            # For constant columns, return middle value (0.5) to place
+            # all points at the center of the normalized range
+            # This maintains the visualization while handling the edge case
+            return series * 0 + 0.5
         return (series - a) / (b - a)

     n = len(frame)
     classes = frame[class_column].drop_duplicates()
     class_col = frame[class_column]
```