# Bug Report: numpy.char.join Strips Null Byte Separators

**Target**: `numpy.char.join`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.join` silently strips null byte characters when used as a separator, producing empty separators instead and corrupting the join operation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=2, max_size=20))
@settings(max_examples=200)
def test_bug_join_null_byte_separator(s):
    result = char.join('\x00', s)
    if isinstance(result, np.ndarray):
        result = result.item()
    expected = '\x00'.join(s)
    assert result == expected
```

**Failing input**: `s='00'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

result = char.join('\x00', 'abc')
if isinstance(result, np.ndarray):
    result = result.item()
expected = '\x00'.join('abc')

print(f"char.join result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
```

Output:
```
char.join result: 'abc'
Expected: 'a\x00b\x00c'
Match: False
```

## Why This Is A Bug

The documentation states that `join` "Calls :meth:`str.join` element-wise". When Python's `str.join` is called with a null byte separator like `'\x00'.join('abc')`, it correctly produces `'a\x00b\x00c'`. However, `numpy.char.join` strips the null bytes, producing just `'abc'`.

## Fix

This is likely related to the same underlying issue affecting `char.add`, `char.multiply`, and `char.replace` - C string operations treating null bytes as terminators. The fix requires properly handling null bytes in the separator string.