# Bug Report: scipy.io.arff split_data_line Crashes on Carriage Return

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with a `csv.Error` when the input contains carriage return characters (`\r`), because the underlying `csv.reader` interprets `\r` as a newline character in unquoted fields.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line


@given(st.just('0\r0'))
def test_split_data_line_carriage_return(line):
    result, dialect = split_data_line(line)
```

**Failing input**: `'0\r0'`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

result, dialect = split_data_line('0\r0')
```

Output:
```
_csv.Error: new-line character seen in unquoted field - do you need to open the file with newline=''?
```

## Why This Is A Bug

Carriage return characters can legitimately appear in ARFF data files, particularly when files are created on Windows systems (which use `\r\n` line endings) or when data values themselves contain `\r` characters. The function already handles newline characters (`\n`) by stripping them, but fails to handle carriage returns. This causes the csv.reader to interpret `\r` as a field delimiter, leading to a crash.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -476,9 +476,11 @@ def split_data_line(line, dialect=None):
     if line[-1] == '\n':
         line = line[:-1]

-    # Remove potential trailing whitespace
+    # Remove carriage returns and potential trailing whitespace
+    line = line.replace('\r', '')
     line = line.strip()

     sniff_line = line
```