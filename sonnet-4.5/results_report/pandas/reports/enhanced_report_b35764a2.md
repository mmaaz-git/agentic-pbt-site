# Bug Report: pandas.from_dummies fails with empty DataFrame from get_dummies(drop_first=True)

**Target**: `pandas.core.reshape.encoding.from_dummies`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `from_dummies` function fails to correctly invert the operation of `get_dummies(..., drop_first=True)` when the input DataFrame has categorical columns with only one unique value each, resulting in an empty dummy-encoded DataFrame that cannot be decoded despite having all necessary information in the `default_category` parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd


@given(
    df=data_frames(
        columns=[
            column("A", elements=st.sampled_from(["cat", "dog", "bird"])),
            column("B", elements=st.sampled_from(["x", "y", "z"])),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )
)
@settings(max_examples=200)
def test_get_dummies_from_dummies_with_drop_first(df):
    """
    Property: from_dummies should invert get_dummies(drop_first=True).
    Evidence: encoding.py line 376 states from_dummies "Inverts the operation
    performed by :func:`~pandas.get_dummies`."
    """
    dummies = pd.get_dummies(df, drop_first=True, dtype=int)

    default_cats = {}
    for col in df.columns:
        first_val = sorted(df[col].unique())[0]
        default_cats[col] = first_val

    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)

    pd.testing.assert_frame_equal(recovered, df)

if __name__ == "__main__":
    test_get_dummies_from_dummies_with_drop_first()
```

<details>

<summary>
**Failing input**: `df = pd.DataFrame({"A": ["cat"], "B": ["x"]})`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 34, in <module>
    test_get_dummies_from_dummies_with_drop_first()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 7, in test_get_dummies_from_dummies_with_drop_first
    df=data_frames(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 29, in test_get_dummies_from_dummies_with_drop_first
    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/encoding.py", line 524, in from_dummies
    raise ValueError(len_msg)
ValueError: Length of 'default_category' (2) did not match the length of the columns being encoded (0)
Falsifying example: test_get_dummies_from_dummies_with_drop_first(
    df=
             A  B
        0  cat  x
    ,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Create a minimal DataFrame with single unique values per column
df = pd.DataFrame({"A": ["cat"], "B": ["x"]})
print("Original DataFrame:")
print(df)
print()

# Apply get_dummies with drop_first=True
dummies = pd.get_dummies(df, drop_first=True, dtype=int)
print("Result from get_dummies(drop_first=True):")
print(dummies)
print(f"Shape: {dummies.shape}")
print()

# Try to reconstruct using from_dummies
default_cats = {"A": "cat", "B": "x"}
print(f"Attempting from_dummies with default_category={default_cats}")
print()

try:
    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)
    print("Successfully recovered:")
    print(recovered)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError when attempting to decode empty DataFrame
</summary>
```
Original DataFrame:
     A  B
0  cat  x

Result from get_dummies(drop_first=True):
Empty DataFrame
Columns: []
Index: [0]
Shape: (1, 0)

Attempting from_dummies with default_category={'A': 'cat', 'B': 'x'}

Error occurred: ValueError: Length of 'default_category' (2) did not match the length of the columns being encoded (0)
```
</details>

## Why This Is A Bug

This is a clear violation of the documented contract between `get_dummies` and `from_dummies`. The documentation at line 376 of `pandas/core/reshape/encoding.py` explicitly states that `from_dummies` "Inverts the operation performed by :func:`~pandas.get_dummies`" without any caveats or exceptions.

The specific issue occurs when:
1. **Input condition**: A DataFrame where each categorical column has exactly one unique value
2. **Operation**: `get_dummies(..., drop_first=True)` is applied, which drops the first dummy column for each categorical variable
3. **Result**: Since there's only one unique value per column, all dummy columns are dropped, resulting in an empty DataFrame (shape: (n_rows, 0))
4. **Problem**: `from_dummies` cannot process this empty DataFrame even though the `default_category` parameter contains all the information needed to reconstruct the original data

The bug is in the validation logic at line 518 of `encoding.py`. When the dummy-encoded DataFrame has no columns (`variables_slice` is empty), the function still tries to validate that the length of `default_category` matches the length of `variables_slice` (which is 0), causing the ValueError. However, in this edge case, the function should recognize that an empty DataFrame with provided `default_category` values can still be decoded - all rows should simply use the default categories.

## Relevant Context

This bug affects a legitimate use case in data processing pipelines:
- **Statistical modeling**: Using `drop_first=True` is a standard practice to avoid the dummy variable trap in regression models
- **Data filtering**: When filtering datasets, subsets can naturally end up with single unique values per categorical column
- **Round-trip expectation**: Users reasonably expect that encoding and decoding operations should be invertible as documented

The `drop_first` parameter is documented in `get_dummies` (line 82-84) as: "Whether to get k-1 dummies out of k categorical levels by removing the first level." The function correctly implements this by returning an empty DataFrame when there's only one level (lines 295-296).

The `default_category` parameter in `from_dummies` (lines 390-394) is specifically designed to handle cases where values need to be assigned when no dummy columns are set to 1, making it the perfect mechanism to handle this edge case.

## Proposed Fix

```diff
--- a/pandas/core/reshape/encoding.py
+++ b/pandas/core/reshape/encoding.py
@@ -515,12 +515,19 @@ def from_dummies(

     if default_category is not None:
         if isinstance(default_category, dict):
-            if not len(default_category) == len(variables_slice):
+            # Handle empty DataFrame case when all columns were dropped
+            if len(variables_slice) == 0 and len(data.columns) == 0:
+                # Reconstruct using only default_category
+                result = DataFrame(index=data.index)
+                for key, value in default_category.items():
+                    result[key] = value
+                return result
+            elif not len(default_category) == len(variables_slice):
                 len_msg = (
                     f"Length of 'default_category' ({len(default_category)}) "
                     f"did not match the length of the columns being encoded "
                     f"({len(variables_slice)})"
                 )
                 raise ValueError(len_msg)
         elif isinstance(default_category, Hashable):
             default_category = dict(
                 zip(variables_slice, [default_category] * len(variables_slice))
```