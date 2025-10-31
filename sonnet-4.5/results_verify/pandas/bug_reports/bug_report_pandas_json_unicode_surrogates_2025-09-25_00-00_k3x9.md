# Bug Report: pandas.io.json.to_json Unicode Surrogate Crash

**Target**: `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `DataFrame.to_json()` method crashes with a `UnicodeEncodeError` when a DataFrame contains Unicode surrogate characters (U+D800 to U+DFFF), rather than handling them gracefully or escaping them.

## Property-Based Test

```python
@given(
    data_frames(
        columns=[
            column("a", dtype=int),
            column("b", dtype=str),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )
)
@settings(max_examples=200)
def test_table_orient_round_trip(df):
    json_str = df.to_json(orient="table")
    parsed = json.loads(json_str)
    assert "schema" in parsed
    assert "data" in parsed
    df_recovered = pd.read_json(StringIO(json_str), orient="table")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df_recovered.reset_index(drop=True))
```

**Failing input**: DataFrame with column containing `'\ud800'` (Unicode surrogate character)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'a': [0],
    'b': ['\ud800']
})

json_str = df.to_json(orient="table")
```

**Output:**
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

Unicode surrogate characters (U+D800 to U+DFFF) are valid in Python strings (they're used internally for UTF-16 encoding of characters outside the Basic Multilingual Plane). While they're not valid UTF-8 characters on their own, libraries should handle them gracefully rather than crashing.

The crash occurs because pandas uses `ujson_dumps()` which attempts to encode the string to UTF-8, and UTF-8 encoding doesn't allow unpaired surrogates.

Expected behaviors:
1. Replace surrogates with replacement character (U+FFFD)
2. Use JSON unicode escaping (`\uD800`)
3. Raise a clear ValueError explaining the issue
4. Add a parameter to control surrogate handling

Instead, it crashes with an unclear UnicodeEncodeError deep in the stack.

## Fix

The fix should be in the JSON encoding layer. One approach is to use `errors='surrogatepass'` or `errors='replace'` when encoding to UTF-8, or to escape surrogates before encoding:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -260,7 +260,12 @@ class JSONTableWriter(FrameWriter):

     def write(self) -> str | None:
-        return ujson_dumps(
+        try:
+            return ujson_dumps(
+                obj, ensure_ascii=self.ensure_ascii, ...
+            )
+        except UnicodeEncodeError:
+            # Handle surrogates by escaping them
+            return ujson_dumps(
-            obj, ensure_ascii=self.ensure_ascii, ...
+            obj, ensure_ascii=True, ...  # Force ASCII escaping
         )
```

Alternatively, preprocess strings to escape or replace surrogates before passing to ujson_dumps.