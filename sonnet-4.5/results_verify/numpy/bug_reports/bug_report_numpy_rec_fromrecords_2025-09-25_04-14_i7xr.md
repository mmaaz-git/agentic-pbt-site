# Bug Report: numpy.rec.fromrecords Null Character Truncation

**Target**: `numpy.rec.fromrecords`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `numpy.rec.fromrecords` processes text data containing strings with trailing null characters (`\x00`), it silently truncates them, causing data corruption. Null characters are only preserved when followed by non-null characters.

## Property-Based Test

```python
import numpy.rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.tuples(st.integers(), st.text(max_size=10)), min_size=1, max_size=20))
@settings(max_examples=500)
def test_fromrecords_dtype_inference(records):
    r = numpy.rec.fromrecords(records, names='num,text')

    assert len(r) == len(records)
    assert r.dtype.names == ('num', 'text')

    for i, (expected_num, expected_text) in enumerate(records):
        assert r.num[i] == expected_num
        assert r.text[i] == expected_text
```

**Failing input**: `records=[(0, '\x00')]`

## Reproducing the Bug

```python
import numpy.rec

test_cases = [
    '\x00',
    'a\x00',
    '\x00b',
    'a\x00b',
]

for s in test_cases:
    r = numpy.rec.fromrecords([(s,)], names='text')
    print(f"Input: {repr(s):10} → Output: {repr(str(r.text[0])):10} | Preserved: {r.text[0] == s}")
```

Output:
```
Input: '\x00'      → Output: ''         | Preserved: False
Input: 'a\x00'     → Output: 'a'        | Preserved: False
Input: '\x00b'     → Output: '\x00b'    | Preserved: True
Input: 'a\x00b'    → Output: 'a\x00b'   | Preserved: True
```

The pattern:
- Null characters at the END are truncated
- Null characters with characters AFTER them are preserved
- This inconsistency affects both `numpy.rec.fromrecords` and base `numpy.array`

## Why This Is A Bug

The function exhibits inconsistent null character handling: preserving them when followed by other characters but truncating them when at the end of strings. This silently corrupts input data and violates the expectation that record array construction preserves input values. The inconsistency indicates unintentional behavior rather than a documented feature.

## Fix

This is a fundamental issue in NumPy's Unicode string dtype handling (affects `np.array` as well, not just `numpy.rec`). The bug stems from C-style string handling where null characters act as terminators. NumPy's Unicode dtype should treat null as a valid Unicode code point. The fix requires updating NumPy's core string dtype implementation to consistently preserve all Unicode characters, including trailing nulls.