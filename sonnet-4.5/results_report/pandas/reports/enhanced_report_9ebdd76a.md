# Bug Report: pandas.io.json String Column Names Converted to Integers on Round-Trip

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

String column names and index values that represent numbers (e.g., '0', '123') are silently converted to integers during JSON serialization/deserialization round-trip, breaking data integrity and causing KeyError when accessing columns by their original string names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas as pd
from io import StringIO
from pandas.testing import assert_frame_equal

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text()),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'columns']),
)
@example(data=[{'0': 0}], orient='records')
def test_read_json_to_json_roundtrip(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)
    assert_frame_equal(df, df_back)

if __name__ == "__main__":
    # Run with the specific failing example
    test_read_json_to_json_roundtrip()
```

<details>

<summary>
**Failing input**: `data=[{'0': 0}], orient='records'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/23
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo_simple.py::test_read_json_to_json_roundtrip FAILED                  [100%]

=================================== FAILURES ===================================
_______________________ test_read_json_to_json_roundtrip _______________________
hypo_simple.py:7: in test_read_json_to_json_roundtrip
    data=st.lists(
               ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo_simple.py:24: in test_read_json_to_json_roundtrip
    assert_frame_equal(df, df_back)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:236: in _check_types
    assert_attr_equal("inferred_type", left, right, obj=obj)
E   AssertionError: DataFrame.columns are different
E
E   Attribute "inferred_type" are different
E   [left]:  string
E   [right]: integer
E   Falsifying explicit example: test_read_json_to_json_roundtrip(
E       data=[{'0': 0}],
E       orient='records',
E   )
=========================== short test summary info ============================
FAILED hypo_simple.py::test_read_json_to_json_roundtrip - AssertionError: Dat...
============================== 1 failed in 0.36s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Test case that demonstrates the bug
df = pd.DataFrame([{'0': 123}])
print(f"Original columns: {df.columns.tolist()}")
print(f"Column type: {type(df.columns[0])}")

json_str = df.to_json(orient='records')
print(f"JSON: {json_str}")

df_back = pd.read_json(StringIO(json_str), orient='records')
print(f"Round-trip columns: {df_back.columns.tolist()}")
print(f"Column type: {type(df_back.columns[0])}")
print(f"Columns equal? {df.columns.equals(df_back.columns)}")

# Demonstrate the KeyError that results
print("\nTrying to access column by original name:")
try:
    print(f"df['0'] = {df['0'].tolist()}")
except KeyError as e:
    print(f"KeyError on original df: {e}")

try:
    print(f"df_back['0'] = {df_back['0'].tolist()}")
except KeyError as e:
    print(f"KeyError on round-trip df: {e}")

# Show how to access it after round-trip
print(f"df_back[0] = {df_back[0].tolist()}")
```

<details>

<summary>
Column names change from string '0' to integer 0, causing KeyError
</summary>
```
Original columns: ['0']
Column type: <class 'str'>
JSON: [{"0":123}]
Round-trip columns: [0]
Column type: <class 'numpy.int64'>
Columns equal? False

Trying to access column by original name:
df['0'] = [123]
KeyError on round-trip df: '0'
df_back[0] = [123]
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **JSON Specification Violation**: JSON object keys are always strings by specification. When pandas converts string keys like '0' to integer 0, it's changing the fundamental data type in a way that contradicts the JSON format itself.

2. **Silent Data Corruption**: The conversion happens silently without any warning. Users have no indication that their column names have been changed, leading to runtime errors when they try to access columns by their original names.

3. **Round-Trip Property Violation**: A fundamental expectation of serialization/deserialization is that data can be round-tripped without loss. The pandas documentation implies this should work, but the default behavior (`convert_axes=True`) breaks this property.

4. **Breaking Code**: Code that works with the original DataFrame will fail with the deserialized one. For example:
   - `df['0']` works but `df_back['0']` raises KeyError
   - Merges and joins using these columns will fail
   - Column-based operations expecting string names will break

5. **Common Use Case**: Numeric-looking string identifiers are extremely common in real data (postal codes, product IDs, year columns like '2024', etc.). This isn't an edge case.

## Relevant Context

The issue stems from the `convert_axes` parameter in `read_json()` which defaults to `True` for most orient values. This triggers the `_convert_axes()` method in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py` (lines 1189-1207) which attempts to convert axes to "proper dtypes".

The conversion logic uses pandas' type inference which aggressively converts numeric-looking strings to integers, even though these were explicitly string keys in the JSON object.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html

The workaround is to explicitly use `pd.read_json(..., convert_axes=False)`, but this requires users to know about the issue beforehand.

## Proposed Fix

The conversion logic should be more conservative and respect that JSON object keys are always strings. When deserializing from orient='records' or orient='index', the column names come from JSON object keys and should remain as strings:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1296,8 +1296,13 @@ class Parser:
             return data, False

         # if we have an index, we want to preserve dtypes
-        if name == "index" and len(data):
-            if self.orient == "split":
+        if name == "index":
+            # For orient='split', preserve exact index from JSON
+            # For other orients, index comes from JSON object keys which are always strings
+            if self.orient == "split":
+                return data, False
+            elif self.orient in ("index", "columns", "records"):
+                # JSON keys are always strings, don't convert
                 return data, False

         return data, converted
```