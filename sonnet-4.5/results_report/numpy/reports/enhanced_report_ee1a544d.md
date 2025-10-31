# Bug Report: numpy.rec.format_parser Unconditionally Strips Whitespace from Field Names

**Target**: `numpy.rec.format_parser`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_parser` class strips whitespace from field names even when names are provided as a list (not comma-separated string), causing field names to be silently modified and whitespace-only names to become empty strings.

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

if __name__ == "__main__":
    test_format_parser_name_count_matches()
```

<details>

<summary>
**Failing input**: `names=[' '], formats=['i4']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 20, in <module>
    test_format_parser_name_count_matches()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 6, in test_format_parser_name_count_matches
    st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=',')), min_size=1, max_size=5),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 17, in test_format_parser_name_count_matches
    assert parser.dtype.names[i] == name  # FAILS when name is '\r', '\n', etc.
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_format_parser_name_count_matches(
    names=[' '],
    formats=['i4'],  # or any other generated value
)
```
</details>

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

<details>

<summary>
Output showing field name '\r' becomes empty string ''
</summary>
```
Input name: '\r'
Actual name: ''
Cannot access with '\r': FAILED
Can access with '': SUCCESS
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Silent Modification**: When a user provides explicit field names as a list like `[' field', '\t']`, they reasonably expect those exact names to be used. The current implementation silently modifies these names without warning.

2. **NumPy Supports Whitespace in Field Names**: NumPy's dtype system explicitly supports field names containing whitespace. For example, `np.array([(1,)], dtype=[(' field', 'i4')])` works correctly and the field can be accessed with `arr[' field']`. The `format_parser` stripping behavior contradicts this capability.

3. **Undocumented Behavior**: The NumPy documentation for `format_parser` does not mention that whitespace will be stripped from field names. Users cannot predict this behavior from the documentation.

4. **Inconsistent Input Format Handling**: While stripping makes sense for comma-separated strings like `'field1, field2'` (to handle spaces after commas), it's unexpected when names are provided as an explicit list where each element is deliberately specified.

5. **Data Loss**: Whitespace-only names like `'\r'`, `'\n'`, `'\t'`, or `' '` become empty strings, potentially causing naming collisions and making fields inaccessible by their original names.

## Relevant Context

The issue occurs in `/numpy/_core/records.py` at line 159 in the `_setfieldnames` method:

```python
def _setfieldnames(self, names, titles):
    """convert input field names into a list and assign to the _names
    attribute """

    if names:
        if type(names) in [list, tuple]:
            pass
        elif isinstance(names, str):
            names = names.split(',')
        else:
            raise NameError(f"illegal input names {repr(names)}")

        self._names = [n.strip() for n in names[:self._nfields]]  # Line 159 - unconditional stripping
```

The code unconditionally applies `.strip()` to all field names regardless of whether they came from a comma-separated string (where stripping makes sense) or from a list/tuple (where it doesn't).

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.rec.format_parser.html

## Proposed Fix

Only strip whitespace when parsing comma-separated strings, not when names are provided as a list:

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