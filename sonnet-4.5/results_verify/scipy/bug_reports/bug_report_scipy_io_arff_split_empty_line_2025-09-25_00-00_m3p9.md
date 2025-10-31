# Bug Report: scipy.io.arff split_data_line Empty String Crash

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an `IndexError` when passed an empty string. This bug is triggered when parsing relational attributes whose data ends with a newline character, as the `.split("\n")` operation produces an empty string that is then passed to `split_data_line`.

## Property-Based Test

```python
from io import StringIO

from hypothesis import given, settings, strategies as st
from scipy.io import arff


@given(
    has_trailing_newline=st.booleans(),
    num_rows=st.integers(min_value=1, max_value=10),
    values=st.data()
)
@settings(max_examples=200)
def test_relational_attribute_handles_trailing_newline(has_trailing_newline, num_rows, values):
    rows = []
    for _ in range(num_rows):
        val = values.draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
        rows.append(str(val))

    relational_data = '\\n'.join(rows)
    if has_trailing_newline:
        relational_data += '\\n'

    content = f"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"{relational_data}"
"""

    f = StringIO(content)
    data, meta = arff.loadarff(f)

    assert len(data) == 1
```

**Failing input**: `has_trailing_newline=True, num_rows=1, values containing 0.0`

## Reproducing the Bug

```python
from io import StringIO

from scipy.io import arff

content = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"0.0\n"
"""

f = StringIO(content)
data, meta = arff.loadarff(f)
```

This raises:
```
IndexError: string index out of range
```

## Why This Is A Bug

In `scipy/io/arff/_arffread.py` at line 476, `split_data_line` attempts to access `line[-1]` without first checking if the line is empty:

```python
def split_data_line(line, dialect=None):
    # ...
    if line[-1] == '\n':  # IndexError if line is empty!
        line = line[:-1]
```

When `RelationalAttribute.parse_data` (line 373) processes relational data, it splits the data string on newlines:

```python
for raw in escaped_string.split("\n"):
    row, self.dialect = split_data_line(raw, self.dialect)
```

When `escaped_string` ends with `\n`, the `split("\n")` produces an empty string as the last element, which then crashes `split_data_line`.

This affects any ARFF file with relational attributes where the relational data ends with a newline character - a common occurrence when data is generated programmatically or includes trailing whitespace.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -473,7 +473,7 @@ def split_data_line(line, dialect=None):
     csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

     # Remove the line end if any
-    if line[-1] == '\n':
+    if line and line[-1] == '\n':
         line = line[:-1]

     # Remove potential trailing whitespace
```

Alternatively, filter empty strings in `RelationalAttribute.parse_data`:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -370,7 +370,9 @@ class RelationalAttribute(Attribute):
         escaped_string = data_str.encode().decode("unicode-escape")

         row_tuples = []

         for raw in escaped_string.split("\n"):
+            if not raw:
+                continue
             row, self.dialect = split_data_line(raw, self.dialect)
```