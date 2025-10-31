# Bug Report: pandas.io.json build_table_schema produces invalid Table Schema with boolean primary_key

**Target**: `pandas.io.json.build_table_schema`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `build_table_schema` function violates the Table Schema specification when called with `primary_key=True` or `primary_key=False`, producing invalid JSON with boolean `primaryKey` values instead of the required array of field names.

## Property-Based Test

```python
import pandas as pd
import pandas.io.json as pj
from hypothesis import given, assume, settings, strategies as st

@given(st.data())
@settings(max_examples=50)
def test_build_table_schema_primary_key_type(data):
    # Generate a random DataFrame with dictionary data
    df_dict = data.draw(st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=5
    ))

    # Ensure all lists have the same length (required for DataFrame)
    assume(all(len(v) == len(list(df_dict.values())[0]) for v in df_dict.values()))

    df = pd.DataFrame(df_dict)

    # Test with different primary_key values
    for pk_value in [None, True, False]:
        schema = pj.build_table_schema(df, index=True, primary_key=pk_value)

        # According to Table Schema spec, primaryKey should be either:
        # - a string (single field name)
        # - an array of strings (multiple field names)
        # - not present
        if 'primaryKey' in schema:
            pk = schema['primaryKey']
            assert isinstance(pk, (list, type(None))), \
                f"primaryKey should be list or None per Table Schema spec, got {type(pk)} with value {pk}"

# Run the test
if __name__ == "__main__":
    test_build_table_schema_primary_key_type()
```

<details>

<summary>
**Failing input**: `primary_key=True` with DataFrame `{'a': [0]}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 36, in <module>
    test_build_table_schema_primary_key_type()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_build_table_schema_primary_key_type
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 31, in test_build_table_schema_primary_key_type
    assert isinstance(pk, (list, type(None))), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: primaryKey should be list or None per Table Schema spec, got <class 'bool'> with value True
Falsifying example: test_build_table_schema_primary_key_type(
    data=data(...),
)
Draw 1: {'a': [0]}
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pandas.io.json as pj
import json

# Create a simple DataFrame
df = pd.DataFrame({'a': [1, 2, 3]})

# Test with primary_key=True
schema_true = pj.build_table_schema(df, index=True, primary_key=True)
print("With primary_key=True:")
print(json.dumps(schema_true, indent=2))
print(f"Type of primaryKey: {type(schema_true['primaryKey'])}")
print(f"Value of primaryKey: {schema_true['primaryKey']}")

print("\n" + "="*50 + "\n")

# Test with primary_key=False
schema_false = pj.build_table_schema(df, index=True, primary_key=False)
print("With primary_key=False:")
print(json.dumps(schema_false, indent=2))
print(f"Type of primaryKey: {type(schema_false['primaryKey'])}")
print(f"Value of primaryKey: {schema_false['primaryKey']}")

print("\n" + "="*50 + "\n")

# Test with primary_key=None (default)
schema_none = pj.build_table_schema(df, index=True, primary_key=None)
print("With primary_key=None (default):")
print(json.dumps(schema_none, indent=2))
print(f"Type of primaryKey: {type(schema_none['primaryKey'])}")
print(f"Value of primaryKey: {schema_none['primaryKey']}")
```

<details>

<summary>
Output showing invalid boolean primaryKey values
</summary>
```
With primary_key=True:
{
  "fields": [
    {
      "name": "index",
      "type": "integer"
    },
    {
      "name": "a",
      "type": "integer"
    }
  ],
  "primaryKey": true,
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'bool'>
Value of primaryKey: True

==================================================

With primary_key=False:
{
  "fields": [
    {
      "name": "index",
      "type": "integer"
    },
    {
      "name": "a",
      "type": "integer"
    }
  ],
  "primaryKey": false,
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'bool'>
Value of primaryKey: False

==================================================

With primary_key=None (default):
{
  "fields": [
    {
      "name": "index",
      "type": "integer"
    },
    {
      "name": "a",
      "type": "integer"
    }
  ],
  "primaryKey": [
    "index"
  ],
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'list'>
Value of primaryKey: ['index']
```
</details>

## Why This Is A Bug

The function `build_table_schema` is specifically designed to create Table Schema compliant JSON output, as documented in its docstring and evidenced by the reference to the Table Schema specification at the top of the module. The Table Schema specification (https://specs.frictionlessdata.io/table-schema/) explicitly defines `primaryKey` as either a string or an array of strings representing field names that uniquely identify a row.

When `primary_key=True` or `primary_key=False` is passed, the function directly assigns these boolean values to the `primaryKey` field in the output schema (line 317 in _table_schema.py), violating the specification. This breaks interoperability with any tool that expects valid Table Schema JSON, including the entire Frictionless Data ecosystem and data validation tools.

The pandas documentation states that `primary_key` accepts "bool or None" as valid parameter types, but the implementation fails to handle boolean values correctly, producing non-compliant output that defeats the function's core purpose.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_table_schema.py` at lines 311-317:

```python
if index and data.index.is_unique and primary_key is None:
    if data.index.nlevels == 1:
        schema["primaryKey"] = [data.index.name]
    else:
        schema["primaryKey"] = data.index.names
elif primary_key is not None:
    schema["primaryKey"] = primary_key  # Bug: directly assigns boolean value
```

The function correctly handles `primary_key=None` by creating a list of index names, but when a boolean is passed, it's directly assigned without conversion. This is tested with pandas version 2.3.2.

Table Schema documentation: https://specs.frictionlessdata.io/table-schema/#primary-key
Pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.io.json.build_table_schema.html

## Proposed Fix

```diff
--- a/pandas/io/json/_table_schema.py
+++ b/pandas/io/json/_table_schema.py
@@ -311,7 +311,15 @@ def build_table_schema(
     if index and data.index.is_unique and primary_key is None:
         if data.index.nlevels == 1:
             schema["primaryKey"] = [data.index.name]
         else:
             schema["primaryKey"] = data.index.names
-    elif primary_key is not None:
-        schema["primaryKey"] = primary_key
+    elif primary_key is not None:
+        if isinstance(primary_key, bool):
+            if primary_key and index:
+                # primary_key=True: use index as primary key
+                if data.index.nlevels == 1:
+                    schema["primaryKey"] = [data.index.name]
+                else:
+                    schema["primaryKey"] = data.index.names
+            # primary_key=False: don't add primaryKey field
+        else:
+            # Assume it's a list of column names
+            schema["primaryKey"] = primary_key
```