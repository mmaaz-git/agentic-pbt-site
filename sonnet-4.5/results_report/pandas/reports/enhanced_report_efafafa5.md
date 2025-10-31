# Bug Report: pandas.io.json Empty DataFrame Index Type Lost During Round-Trip

**Target**: `pandas.io.json._json` (specifically `read_json` and `to_json` with `orient='split'`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When serializing an empty DataFrame to JSON with `orient='split'` and reading it back, the index type is not preserved. A `RangeIndex` is incorrectly converted to a regular `Index` with `float64` dtype.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import tempfile
import os


@given(
    data_frames(
        columns=[
            column("int_col", dtype=int),
            column("float_col", dtype=float),
            column("str_col", dtype=str),
        ],
        index=range_indexes(min_size=0, max_size=100),
    )
)
@settings(max_examples=100)
def test_json_round_trip_orient_split(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        df.to_json(temp_path, orient="split")
        result = pd.read_json(temp_path, orient="split")
        pd.testing.assert_frame_equal(df, result)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run the test
    test_json_round_trip_orient_split()
```

<details>

<summary>
**Failing input**: `Empty DataFrame with columns [int_col, float_col, str_col]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 34, in <module>
    test_json_round_trip_orient_split()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 9, in test_json_round_trip_orient_split
    data_frames(
               ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 26, in test_json_round_trip_orient_split
    pd.testing.assert_frame_equal(df, result)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1250, in assert_frame_equal
    assert_index_equal(
    ~~~~~~~~~~~~~~~~~~^
        left.index,
        ^^^^^^^^^^^
    ...<8 lines>...
        obj=f"{obj}.index",
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 253, in assert_index_equal
    _check_types(left, right, obj=obj)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 236, in _check_types
    assert_attr_equal("inferred_type", left, right, obj=obj)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.index are different

Attribute "inferred_type" are different
[left]:  integer
[right]: floating
Falsifying example: test_json_round_trip_orient_split(
    df=
        Empty DataFrame
        Columns: [int_col, float_col, str_col]
        Index: []
    ,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import os

# Create an empty DataFrame with specific columns
df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])
print(f"Original index type: {type(df.index)}")
print(f"Original index: {df.index}")
print(f"Original index inferred_type: {df.index.inferred_type}")
print()

# Save to JSON with orient='split'
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

try:
    df.to_json(temp_path, orient="split")

    # Read back from JSON
    result = pd.read_json(temp_path, orient="split")

    print(f"Result index type: {type(result.index)}")
    print(f"Result index: {result.index}")
    print(f"Result index inferred_type: {result.index.inferred_type}")
    print()
    print(f"Index types match: {type(df.index) == type(result.index)}")
    print(f"Index inferred_types match: {df.index.inferred_type == result.index.inferred_type}")

    # Try to verify they are equal
    print("\nTrying pd.testing.assert_frame_equal:")
    try:
        pd.testing.assert_frame_equal(df, result)
        print("DataFrames are equal")
    except AssertionError as e:
        print(f"AssertionError: {e}")

finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

<details>

<summary>
Empty DataFrame index type changes from RangeIndex to float64 Index
</summary>
```
Original index type: <class 'pandas.core.indexes.range.RangeIndex'>
Original index: RangeIndex(start=0, stop=0, step=1)
Original index inferred_type: integer

Result index type: <class 'pandas.core.indexes.base.Index'>
Result index: Index([], dtype='float64')
Result index inferred_type: floating

Index types match: False
Index inferred_types match: False

Trying pd.testing.assert_frame_equal:
AssertionError: DataFrame.index are different

Attribute "inferred_type" are different
[left]:  integer
[right]: floating
```
</details>

## Why This Is A Bug

This violates the expected behavior of JSON round-trip serialization with `orient='split'` for several reasons:

1. **Documentation Contract Violation**: The pandas documentation explicitly states that `orient='split'` enables "round-trip conversion" and "preserves original DataFrame structure". An index is a fundamental part of DataFrame structure.

2. **Inconsistent Behavior**: Non-empty DataFrames preserve their index structure (though RangeIndex becomes regular Index, the dtype is preserved), but empty DataFrames lose both the index type AND switch to an unexpected float64 dtype. This inconsistency makes the behavior unpredictable.

3. **pandas' Own Testing Framework Fails**: The `pd.testing.assert_frame_equal()` function, which is pandas' official way to compare DataFrames, considers these DataFrames unequal specifically due to the index type difference (`inferred_type: integer` vs `floating`).

4. **Performance and Type Safety Impact**: RangeIndex is a memory-efficient index type in pandas. Users who filter DataFrames to empty sets and then serialize them lose this optimization. Additionally, type-checking code that expects integer-based indexes will fail unexpectedly.

## Relevant Context

The issue occurs in the `_try_convert_data` method in `/pandas/io/json/_json.py`. When `convert_axes=True` (the default for `orient='split'`), the code attempts to convert axes to "proper dtypes". However, there's a special case at line 1298-1300:

```python
# if we have an index, we want to preserve dtypes
if name == "index" and len(data):
    if self.orient == "split":
        return data, False
```

The problem is the `len(data)` check - for empty indexes, `len(data)` is 0 (False), so this preservation logic is skipped. The empty index then gets converted through the default conversion path, which results in a float64 Index.

Related pandas issues:
- Issue #21287: Similar round-trip issues with empty DataFrames in `orient='table'` (fixed)
- Issue #28558: Acknowledged edge case with empty frames and `convert_axes`

## Proposed Fix

The fix involves removing the length check for index preservation when using `orient='split'`:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1295,8 +1295,9 @@ class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
             except (TypeError, ValueError):
                 pass

-        # if we have an index, we want to preserve dtypes
-        if name == "index" and len(data):
+        # if we have an index, we want to preserve dtypes
+        # This should apply even for empty indexes to maintain consistency
+        if name == "index":
             if self.orient == "split":
                 return data, False
```

This ensures that index types are preserved consistently for both empty and non-empty DataFrames when using `orient='split'`, maintaining the promised round-trip capability.