# Bug Report: scipy.io.arff NominalAttribute Value Parsing

**Target**: `scipy.io.arff._arffread.NominalAttribute._get_nom_val`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `NominalAttribute._get_nom_val()` function incorrectly handles nominal attribute values that contain only whitespace or special CSV characters like quotes. Whitespace-only values are completely lost, and quote characters are stripped, causing data corruption when parsing ARFF nominal attribute definitions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io.arff._arffread import NominalAttribute


@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@settings(max_examples=1000)
def test_nominal_attribute_get_nom_val_round_trip(values):
    nominal_str = "{" + ", ".join(values) + "}"

    try:
        result = NominalAttribute._get_nom_val(nominal_str)
    except ValueError:
        return

    assert len(result) == len(values)
    for i, val in enumerate(values):
        assert result[i] == val
```

**Failing inputs**:
- `values=[' ']` - single space value returns empty tuple
- `values=['"']` - quote character returns empty string

## Reproducing the Bug

```python
from scipy.io.arff._arffread import NominalAttribute

result1 = NominalAttribute._get_nom_val("{ }")
print(f"Space value: {repr(result1)}")

result2 = NominalAttribute._get_nom_val('{"}'  )
print(f"Quote value: {repr(result2)}")

assert result1 == (' ',), f"Expected (' ',) but got {repr(result1)}"
assert result2 == ('"',), f"Expected ('\"',) but got {repr(result2)}"
```

## Why This Is A Bug

The ARFF format allows nominal attributes to have any string value, including whitespace-only strings. The function `_get_nom_val()` uses `split_data_line()` internally, which was designed for parsing ARFF data rows (where stripping whitespace is appropriate). However, when reused for parsing nominal attribute definitions, the `line.strip()` call at line 480 incorrectly removes significant whitespace from attribute values.

This violates the documented behavior that nominal attributes can contain any values specified in the braces `{}`, and causes data corruption when legitimate whitespace values are present in the ARFF file.

## Fix

The root cause is in `split_data_line()` at line 480 in `_arffread.py`, which aggressively strips all whitespace:

```python
line = line.strip()  # Line 480
```

This line should be modified to only remove the trailing newline (which is already handled above at line 476-477), or the function should accept a parameter to control stripping behavior for different use cases.

**Recommended fix:**

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -476,9 +476,6 @@ def split_data_line(line, dialect=None):
     if line[-1] == '\n':
         line = line[:-1]

-    # Remove potential trailing whitespace
-    line = line.strip()
-
     sniff_line = line

     # Add a delimiter if none is present, so that the csv.Sniffer
```

Alternatively, add a parameter to control the behavior:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -468,7 +468,7 @@ def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
         dialect.skipinitialspace = space


-def split_data_line(line, dialect=None):
+def split_data_line(line, dialect=None, strip_whitespace=True):
     delimiters = ",\t"

     # This can not be done in a per reader basis, and relational fields
@@ -478,7 +478,8 @@ def split_data_line(line, dialect=None):
     if line[-1] == '\n':
         line = line[:-1]

-    line = line.strip()
+    if strip_whitespace:
+        line = line.strip()

     sniff_line = line

@@ -128,7 +129,7 @@ class NominalAttribute(Attribute):
         """
         m = r_nominal.match(atrv)
         if m:
-            attrs, _ = split_data_line(m.group(1))
+            attrs, _ = split_data_line(m.group(1), strip_whitespace=False)
             return tuple(attrs)
         else:
             raise ValueError("This does not look like a nominal string")
```