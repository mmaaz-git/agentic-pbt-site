# Bug Report: numpy.rec.format_parser Strips Whitespace from Field Names

**Target**: `numpy.rec.format_parser`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_parser` class strips whitespace from field names even when names are provided as a list (not comma-separated string), leading to unexpected field names. Whitespace-only names like `'\r'`, `' '`, `'\t'` are converted to empty strings, causing inconsistency between user input and actual field names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy.rec as rec


@given(
    st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=',')), min_size=1, max_size=5),
    st.lists(st.sampled_from(['i4', 'f8', 'i2']), min_size=1, max_size=5)
)
def test_format_parser_name_count_matches(names, formats):
    assume(len(names) == len(formats))
    assume(len(set(names)) == len(names))

    parser = rec.format_parser(formats, names, [])

    assert len(parser.dtype.names) == len(formats)
    for i, name in enumerate(names):
        assert parser.dtype.names[i] == name  # FAILS when name is '\r', '\n', etc.
```

**Failing input**: `names=['\r'], formats=['i4']`

## Reproducing the Bug

```python
import numpy.rec as rec

names = ['\r']
formats = ['i4']

parser = rec.format_parser(formats, names, [])

print(f"Input name: {repr(names[0])}")
print(f"Actual name: {repr(parser.dtype.names[0])}")

try:
    field = parser.dtype.fields['\r']
    print("Can access with '\\r': SUCCESS")
except KeyError:
    print("Cannot access with '\\r': FAILED")

try:
    field = parser.dtype.fields['']
    print("Can access with '': SUCCESS")
except KeyError:
    print("Cannot access with '': FAILED")
```

**Output:**
```
Input name: '\r'
Actual name: ''
Cannot access with '\r': FAILED
Can access with '': SUCCESS
```

## Why This Is A Bug

The `format_parser._setfieldnames` method (line 159 in records.py) unconditionally strips all field names:

```python
self._names = [n.strip() for n in names[:self._nfields]]
```

While this makes sense for comma-separated string input like `'field1, field2'` (where spaces after commas should be removed), it's unexpected when names are provided as a list where each name is explicitly specified.

This causes:
1. **Contract violation**: User provides name `'\r'` but gets `''`
2. **Inaccessibility**: Field cannot be accessed with the original name
3. **Inconsistency**: Different behavior based on input format (list vs string)
4. **Empty names**: Whitespace-only names become empty strings

Example showing inconsistency:
```python
r1 = rec.fromarrays([np.array([1,2,3])], names=[' field'])  # User expects ' field'
r2 = rec.fromarrays([np.array([1,2,3])], names=' field')    # User expects trimmed

print(r1.dtype.names)  # ('field',) - same as r2
print(r2.dtype.names)  # ('field',) - trimming makes sense here
```

## Fix

Only strip names when they come from a comma-separated string, not when provided as a list:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -148,11 +148,14 @@ class format_parser:
         """convert input field names into a list and assign to the _names
         attribute """

+        strip_whitespace = False
         if names:
             if type(names) in [list, tuple]:
                 pass
             elif isinstance(names, str):
                 names = names.split(',')
+                # Only strip when parsing comma-separated strings
+                strip_whitespace = True
             else:
                 raise NameError(f"illegal input names {repr(names)}")

-            self._names = [n.strip() for n in names[:self._nfields]]
+            self._names = [n.strip() if strip_whitespace else n for n in names[:self._nfields]]
         else:
             self._names = []
```