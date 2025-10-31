# Bug Report: numpy.rec.format_parser Inconsistent Whitespace Handling

**Target**: `numpy.rec.format_parser`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`format_parser` crashes with cryptic errors (TypeError or SyntaxError) when format strings have leading or trailing whitespace, despite handling whitespace around commas correctly.

## Property-Based Test

```python
import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.sampled_from(['i4', 'f8', 'S10']), min_size=1, max_size=5))
def test_format_parser_extra_spaces_in_format_string(formats):
    names = [f'f{i}' for i in range(len(formats))]

    formats_no_space = ','.join(formats)
    formats_leading_space = ' ' + ','.join(formats)

    parser_no_space = numpy.rec.format_parser(formats_no_space, names, [])
    parser_leading = numpy.rec.format_parser(formats_leading_space, names, [])
    assert parser_no_space.dtype == parser_leading.dtype
```

**Failing input**: `formats=['i4']` (or any format list)

## Reproducing the Bug

```python
import numpy.rec

formats_no_space = 'i4,i4'
formats_leading = ' i4,i4'
names = ['f0', 'f1']

parser1 = numpy.rec.format_parser(formats_no_space, names, [])
print(f"Without leading space: {parser1.dtype}")

parser2 = numpy.rec.format_parser(formats_leading, names, [])
print(f"With leading space: {parser2.dtype}")
```

Output:
```
Without leading space: [('f0', '<i4'), ('f1', '<i4')]
Traceback (most recent call last):
  ...
TypeError: data type ' i4' not understood
```

## Why This Is A Bug

The error handling is inconsistent: whitespace around commas is handled correctly (e.g., `'i4 , f8'` works), but leading/trailing whitespace causes crashes. This violates the principle of least surprise and makes the API fragile to minor formatting variations that users might reasonably introduce, especially when building format strings programmatically.

## Fix

Strip leading/trailing whitespace from format strings in the `_parseFormats` method before processing:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -134,7 +134,10 @@ class format_parser:
         if isinstance(formats, list):
             dtype = sb.dtype([('f{}'.format(i), format_)
                             for i, format_ in enumerate(formats)], aligned)
         else:
+            if isinstance(formats, str):
+                formats = formats.strip()
             dtype = sb.dtype(formats, aligned)
```