# Bug Report: pandas.io.json build_table_schema invalid primaryKey type

**Target**: `pandas.io.json.build_table_schema`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `build_table_schema` is called with `primary_key=True` or `primary_key=False`, it produces an invalid Table Schema by setting `primaryKey` to a boolean value instead of an array of column names or omitting it entirely. This violates the Table Schema specification.

## Property-Based Test

```python
@given(st.data())
@settings(max_examples=50)
def test_build_table_schema_primary_key_type(data):
    df_dict = data.draw(st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=5
    ))

    assume(all(len(v) == len(list(df_dict.values())[0]) for v in df_dict.values()))

    df = pd.DataFrame(df_dict)

    for pk_value in [None, True, False]:
        schema = pj.build_table_schema(df, index=True, primary_key=pk_value)

        if 'primaryKey' in schema:
            pk = schema['primaryKey']
            assert isinstance(pk, (list, type(None))), \
                f"primaryKey should be list or None per Table Schema spec, got {type(pk)} with value {pk}"
```

**Failing input**: `primary_key=True` (or `False`)

## Reproducing the Bug

```python
import pandas as pd
import pandas.io.json as pj
import json

df = pd.DataFrame({'a': [1, 2, 3]})

schema_true = pj.build_table_schema(df, index=True, primary_key=True)
print("With primary_key=True:")
print(json.dumps(schema_true, indent=2))
print(f"Type of primaryKey: {type(schema_true['primaryKey'])}")

schema_false = pj.build_table_schema(df, index=True, primary_key=False)
print("\nWith primary_key=False:")
print(json.dumps(schema_false, indent=2))
print(f"Type of primaryKey: {type(schema_false['primaryKey'])}")

schema_none = pj.build_table_schema(df, index=True, primary_key=None)
print("\nWith primary_key=None:")
print(json.dumps(schema_none, indent=2))
print(f"Type of primaryKey: {type(schema_none['primaryKey'])}")
```

Output:
```
With primary_key=True:
{
  "fields": [...],
  "primaryKey": true,
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'bool'>

With primary_key=False:
{
  "fields": [...],
  "primaryKey": false,
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'bool'>

With primary_key=None:
{
  "fields": [...],
  "primaryKey": ["index"],
  "pandas_version": "1.4.0"
}
Type of primaryKey: <class 'list'>
```

## Why This Is A Bug

The Table Schema specification (https://specs.frictionlessdata.io/table-schema/) defines `primaryKey` as "a primary key is an array of field names, whose values together uniquely identify a row." When `primary_key=True` or `primary_key=False` is passed, the function produces `"primaryKey": true` or `"primaryKey": false`, which is not a valid Table Schema. This breaks interoperability with other tools that expect valid Table Schema JSON.

## Fix

The `primary_key` parameter should be validated, and boolean values should be converted to the appropriate Table Schema representation. When `primary_key=False`, the `primaryKey` field should be omitted from the schema. When `primary_key=True`, it should be set to the index columns (similar to `primary_key=None` when the index is unique).

```diff
--- a/pandas/io/json/_table_schema.py
+++ b/pandas/io/json/_table_schema.py
@@ -somewhere in build_table_schema
-    if primary_key is not None:
-        schema["primaryKey"] = primary_key
+    if primary_key is False:
+        # Don't include primaryKey in schema
+        pass
+    elif primary_key is True:
+        # Set to index columns if unique
+        if index and data.index.is_unique:
+            schema["primaryKey"] = [data.index.name or "index"]
+    elif primary_key is None:
+        # Existing logic for None case
+        ...
+    else:
+        # primary_key should be bool or None
+        raise ValueError(f"primary_key must be bool or None, got {type(primary_key)}")
```