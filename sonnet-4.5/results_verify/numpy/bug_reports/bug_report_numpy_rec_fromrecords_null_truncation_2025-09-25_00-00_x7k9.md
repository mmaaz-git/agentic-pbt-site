# Bug Report: numpy.rec.fromrecords Null Character Truncation

**Target**: `numpy.rec.fromrecords`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords()` silently truncates null characters (`\x00`) from string fields, causing data loss when storing strings containing null bytes.

## Property-Based Test

```python
import numpy.rec as rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.tuples(st.integers(), st.text(alphabet=st.characters(min_codepoint=0, max_codepoint=127)), st.floats(allow_nan=False, allow_infinity=False)), min_size=1, max_size=10))
@settings(max_examples=1000)
def test_fromrecords_preserves_string_data(records):
    result = rec.fromrecords(records, names=['a', 'b', 'c'])

    for i, (a, b, c) in enumerate(records):
        assert result[i].b == b, f"String data lost at index {i}: expected {repr(b)}, got {repr(result[i].b)}"
```

**Failing input**: `records=[(0, '\x00', 0.0)]`

## Reproducing the Bug

```python
import numpy.rec as rec

records = [(1, '\x00', 2.0)]
result = rec.fromrecords(records, names=['a', 'b', 'c'])

print("Input:", repr(records[0][1]))
print("Output:", repr(result[0].b))
print("Lengths:", len(records[0][1]), "vs", len(result[0].b))
assert result[0].b == '\x00'
```

Output:
```
Input: '\x00'
Output: np.str_('')
Lengths: 1 vs 0
AssertionError
```

## Why This Is A Bug

Null characters (`\x00`) are valid Unicode codepoints and should be preserved in string fields. The documentation for `fromrecords` states it creates a record array from the input data, with no mention of character sanitization or truncation. Users expect their data to be stored faithfully.

This is a data corruption issue that affects any application storing binary-like data or strings from sources that may contain null bytes (e.g., file paths, network protocols, or embedded data).

## Fix

This bug originates in NumPy's core array creation logic, not in `numpy.rec` itself. When Python strings containing `\x00` are converted to NumPy Unicode arrays, the null character causes string truncation (similar to C string behavior, where `\x00` terminates strings).

The issue occurs in lines 709-714 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/_core/records.py`:

```python
if formats is None and dtype is None:  # slower
    obj = sb.array(recList, dtype=object)
    arrlist = [
        sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])
    ]
```

The `sb.array()` call converts Python strings to NumPy Unicode dtype, which appears to treat `\x00` as a string terminator.

**Workaround**: Users can pre-encode strings with null characters to bytes:
```python
records = [(1, b'\x00', 2.0)]
result = rec.fromrecords(records, names=['a', 'b', 'c'], formats=['i4', 'S1', 'f8'])
```

**Proper fix**: This requires a fix in NumPy's core string-to-array conversion logic to properly handle null characters in Unicode strings. This is beyond the scope of `numpy.rec` and would need to be addressed in the NumPy C implementation layer.