# Bug Report: pandas JSON Round-Trip Converts Maximum Float64 to Infinity

**Target**: `pandas.DataFrame.to_json` / `pandas.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

JSON serialization round-trip (to_json → read_json) silently corrupts the maximum float64 value (1.7976931348623157e+308) by converting it to infinity, causing data loss for valid finite floating-point values.

## Property-Based Test

```python
from hypothesis import given, settings, seed, Phase
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import numpy as np
import io

@given(
    df=data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
    ], index=range_indexes(min_size=0, max_size=50))
)
@settings(
    max_examples=100,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    print_blob=True
)
@seed(0)  # For reproducibility
def test_to_json_read_json_roundtrip(df):
    json_buffer = io.StringIO()
    df.to_json(json_buffer, orient='records')
    json_buffer.seek(0)
    result = pd.read_json(json_buffer, orient='records')

    if len(df) > 0:
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            result.reset_index(drop=True),
            check_dtype=False
        )

if __name__ == "__main__":
    # Run the test
    test_to_json_read_json_roundtrip()
    print("Test completed successfully if no errors above.")
```

<details>

<summary>
**Failing input**: DataFrame containing `np.finfo(np.float64).max` at index 12
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 34, in <module>
    test_to_json_read_json_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 8, in test_to_json_read_json_roundtrip
    df=data_frames([
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 26, in test_to_json_read_json_roundtrip
    pd.testing.assert_frame_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        df.reset_index(drop=True),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        result.reset_index(drop=True),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        check_dtype=False
        ^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.iloc[:, 1] (column name="b") are different

DataFrame.iloc[:, 1] (column name="b") values are different (5.55556 %)
[index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
[left]:  [-4.1561851617952834e-145, -1.367790740375205e-260, -1.175494351e-38, inf, -5.960464477539063e-08, 4.81205365398809e-27, 1.5605241789407732e+16, -1.2523315338074625e-15, 5.126421159223293e+16, 3.2963214284570204e+16, 1.5605241789407732e+16, -2.1102088963624535e-137, 1.7976931348623157e+308, -2.6785693365291813e+146, -2424045664839840.0, 1.5605241789407732e+16, 1.5605241789407732e+16, 1.5605241789407732e+16, 5.989832527316429e+16, -0.016965377309315525, 1.5605241789407732e+16, 4.811574194813066e-182, 1.5605241789407732e+16, 3.439306710062107e+16, -2973177385616858.0, 1.1754943508222875e-38, -3.3829990665991324e+16, 6.280943529838262e+16, 9007199254740992.0, 2.7630505394003064e+16, 1.5605241789407732e+16, 1.5605241789407732e+16, 1.5605241789407732e+16, 1.5605241789407732e+16, -2.344304298832664e+16, -4.951131351115949e+16]
[right]: [-4.156185162e-145, -1.36779074e-260, -1.175494351e-38, nan, -5.9600000000000004e-08, 4.812053654e-27, 1.5605241790000002e+16, -0.0, 5.126421159e+16, 3.296321428e+16, 1.5605241790000002e+16, -2.110208896e-137, inf, -2.6785693369999997e+146, -2424045664839840.0, 1.5605241790000002e+16, 1.5605241790000002e+16, 1.5605241790000002e+16, 5.989832527e+16, -0.0169653773, 1.5605241790000002e+16, 4.8115741950000005e-182, 1.5605241790000002e+16, 3.43930671e+16, -2973177385616858.0, 1.175494351e-38, -3.382999067e+16, 6.28094353e+16, 9007199254740992.0, 2.763050539e+16, 1.5605241790000002e+16, 1.5605241790000002e+16, 1.5605241790000002e+16, 1.5605241790000002e+16, -2.344304299e+16, -4.951131351e+16]
At positional index 3, first diff: inf != nan
Falsifying example: test_to_json_read_json_roundtrip(
    df=
                              a              b
        0  -9223372036854775658 -4.156185e-145
        1  -9223372036854775658 -1.367791e-260
        2  -9223372036854775658  -1.175494e-38
        3  -9223372036854775658            inf
        4  -9223372036854775658  -5.960464e-08
        5  -9223372036854775658   4.812054e-27
        6   4105189644819390461   1.560524e+16
        7  -9223372036854775658  -1.252332e-15
        8  -9223372036854775658   5.126421e+16
        9  -9223372036854775658   3.296321e+16
        10 -9223372036854775658   1.560524e+16
        11 -9223372036854775658 -2.110209e-137
        12 -9223372036854775658  1.797693e+308
        13 -9223372036854775658 -2.678569e+146
        14 -9223372036854775658  -2.424046e+15
        15 -9223372036854775658   1.560524e+16
        16 -9223372036854775658   1.560524e+16
        17 -9223372036854775658   1.560524e+16
        18 -9223372036854775658   5.989833e+16
        19 -9223372036854775658  -1.696538e-02
        20 -9223372036854775658   1.560524e+16
        21 -9223372036854775658  4.811574e-182
        22 -9223372036854775808   1.560524e+16
        23 -9223372036854775658   3.439307e+16
        24 -9223372036854775658  -2.973177e+15
        25 -9223372036854775658   1.175494e-38
        26 -9223372036854775658  -3.382999e+16
        27 -9223372036854775658   6.280944e+16
        28 -9223372036854775658   9.007199e+15
        29 -9223372036854775658   2.763051e+16
        30 -9223372036854775658   1.560524e+16
        31 -9223372036854775658   1.560524e+16
        32 -9223372036854775658   1.560524e+16
        33 -9223372036854775658   1.560524e+16
        34 -9223372036854775658  -2.344304e+16
        35 -9223372036854775658  -4.951131e+16
    ,
)

You can reproduce this example by temporarily adding @reproduce_failure('6.139.1', b'AXicc1RhdGTzsPgxwTpYKP4vo6OY5/8GBijwgLKmMTryaRxWtAyw7A50YHRk1Oj89drc800LO6Ojssbh1D8n7h9zusHoKKFxWFVBd/ZN7i2Mjtwai/sz8/Zyf9rI6Miisa8AYhCjo7SGc/aT2eofJFIYHZk16j/AxMU1nON6hP9cNHJhdOTUcI51+z6ztX86o6OshnNEV7HYvnA5RkdejXuhxTlF30uAZgpr7J8YH2905Uwg0G1AJzE6MmnsEAAaxZU8m9FRRsPZAWYyj0b9+/9gwOjIrrHnWn+MbPW/J4yOQhrOWTOXbRFsB/pISeNwyKWQTYX1exgdOTSc044IvDFZ/gdoC6OjlMbhOB6eBZJr1cFcXkZHBo2FXw4l7g9kLGV0lNSwEIBZJARGohpSnn0xVZHTvgM9AgwHYLCAXcgLZnCDEauGVfkVho2KMx8DAwEYhgwazt47Cu32ffgFAD2udnI=') as a decorator on your test case
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import io

# Create DataFrame with maximum float64 value
df = pd.DataFrame({'value': [np.finfo(np.float64).max]})

print("Original DataFrame:")
print(df)
print(f"Original value: {df['value'].iloc[0]}")
print(f"Is finite? {np.isfinite(df['value'].iloc[0])}")
print(f"Value == max float64? {df['value'].iloc[0] == np.finfo(np.float64).max}")

# Serialize to JSON
json_buffer = io.StringIO()
df.to_json(json_buffer, orient='records')
json_buffer.seek(0)

# Show the JSON representation
json_str = json_buffer.getvalue()
print(f"\nJSON representation: {json_str}")

# Parse back from JSON
json_buffer.seek(0)
result = pd.read_json(json_buffer, orient='records')

print("\nAfter JSON round-trip:")
print(result)
print(f"Round-trip value: {result['value'].iloc[0]}")
print(f"Is finite? {np.isfinite(result['value'].iloc[0])}")
print(f"Values equal? {df['value'].iloc[0] == result['value'].iloc[0]}")

# Check if it's converted to infinity
if np.isinf(result['value'].iloc[0]):
    print("\n⚠️  ERROR: Valid finite float64 value was converted to infinity!")
```

<details>

<summary>
Output showing data corruption
</summary>
```
Original DataFrame:
           value
0  1.797693e+308
Original value: 1.7976931348623157e+308
Is finite? True
Value == max float64? True

JSON representation: [{"value":1.797693135e+308}]

After JSON round-trip:
   value
0    inf
Round-trip value: inf
Is finite? False
Values equal? False

⚠️  ERROR: Valid finite float64 value was converted to infinity!
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Silent Data Corruption**: A valid, finite float64 value (1.7976931348623157e+308) is silently converted to infinity without any warning or error. This is the maximum representable float64 value according to IEEE 754 standard, not an invalid or overflow value.

2. **Violates Round-Trip Property**: The fundamental expectation of serialization is that `deserialize(serialize(data))` should equal `data`. Here, `pd.read_json(df.to_json())` does not preserve the original data, breaking this invariant.

3. **Inconsistent with JSON Specification**: While RFC 7159 allows implementations to set precision limits, it explicitly prohibits Infinity in JSON. The bug creates an infinity value that couldn't even be serialized back to valid JSON.

4. **Loss of Precision**: The default `double_precision=10` parameter truncates the max float64 from full precision to `1.797693135e+308`. When parsed back, this rounded value exceeds the maximum representable float64 and overflows to infinity.

5. **Additional Issue Found**: The Hypothesis test also revealed that original `inf` values in DataFrames become `nan` after round-trip (see index 3 in the test output), indicating broader issues with special float values.

## Relevant Context

The bug occurs in the interaction between `DataFrame.to_json()` and `pd.read_json()`:

1. **Root Cause**: The `to_json()` method uses a default `double_precision=10` parameter (defined in `/pandas/io/json/_json.py:145`), which truncates floating-point values to 10 significant digits.

2. **Precision Loss**: The maximum float64 value `1.7976931348623157e+308` gets truncated to `1.797693135e+308` in the JSON output.

3. **Overflow on Parse**: When `read_json()` parses `1.797693135e+308`, the slight rounding up causes it to exceed the maximum float64 value, resulting in infinity.

4. **Documentation**: The pandas documentation mentions the `double_precision` parameter but doesn't warn about potential data corruption for extreme values. The maximum allowed precision is 15 (still insufficient for exact float64 representation which needs 17 digits).

5. **Use Cases Affected**: Scientific computing, physics simulations, and financial modeling applications that work with extreme values are at risk of silent data corruption.

## Proposed Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -143,7 +143,7 @@ def to_json(
     orient: str | None = None,
     date_format: str = "epoch",
-    double_precision: int = 10,
+    double_precision: int = 17,  # Ensure float64 values round-trip correctly
     force_ascii: bool = True,
     date_unit: str = "ms",
```

Alternative approaches:
1. **Increase default precision**: Change default `double_precision` from 10 to 17 (the minimum needed for exact float64 round-trips according to IEEE 754)
2. **Clamp on read**: Modify `read_json()` to clamp values to `[-sys.float_info.max, sys.float_info.max]` instead of allowing overflow
3. **Add warnings**: Emit a warning when serializing values near float64 limits with insufficient precision
4. **Document limitation**: At minimum, add clear documentation warning users about this edge case and recommending `double_precision=17` for scientific applications