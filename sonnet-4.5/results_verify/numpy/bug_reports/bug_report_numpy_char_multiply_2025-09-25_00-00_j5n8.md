# Bug Report: numpy.char.multiply Strips Trailing Null Bytes

**Target**: `numpy.char.multiply`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.multiply` silently strips trailing null bytes from strings during multiplication, producing incorrect output that differs from Python's native string repetition.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings


@st.composite
def text_ending_with_null(draw):
    prefix = draw(st.text(min_size=1, max_size=10))
    num_nulls = draw(st.integers(min_value=1, max_value=3))
    return prefix + '\x00' * num_nulls


@given(text_ending_with_null(), st.integers(min_value=1, max_value=5))
@settings(max_examples=200)
def test_bug_multiply_strips_trailing_nulls(s, n):
    arr = np.array([s])
    result = char.multiply(arr, n)[0]
    expected = s * n
    assert result == expected
```

**Failing input**: `s='0\x00'`, `n=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

arr = np.array(['hello\x00'])
result = char.multiply(arr, 2)[0]
expected = 'hello\x00' * 2

print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
```

Output:
```
char.multiply result: np.str_('hellohello')
Expected: 'hello\x00hello\x00'
Match: False
```

## Why This Is A Bug

The documentation states that `multiply` returns "string multiple concatenation, element-wise", matching Python's `*` operator for strings. However, trailing null bytes are silently stripped from the result, violating the expected semantics and potentially causing silent data corruption.

## Fix

Similar to the `char.add` bug, this likely stems from C string operations treating null bytes as terminators. The fix requires using length-aware string operations.