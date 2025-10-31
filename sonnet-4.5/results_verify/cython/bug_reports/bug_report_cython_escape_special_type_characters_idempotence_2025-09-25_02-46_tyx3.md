# Bug Report: Cython.Compiler.PyrexTypes._escape_special_type_characters Idempotence Violation

**Target**: `Cython.Compiler.PyrexTypes._escape_special_type_characters`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_escape_special_type_characters` function is not idempotent: escaping ':' once produces '__D', but escaping '__D' again produces '__dunderD', violating the expected property that escape functions should be idempotent.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import Cython.Compiler.PyrexTypes as PT

@given(st.text())
@settings(max_examples=1000)
def test_escape_special_type_characters_idempotence(s):
    escaped_once = PT._escape_special_type_characters(s)
    escaped_twice = PT._escape_special_type_characters(escaped_once)
    assert escaped_once == escaped_twice
```

**Failing input**: `':'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Compiler.PyrexTypes as PT

s = ':'
escaped_once = PT._escape_special_type_characters(s)
escaped_twice = PT._escape_special_type_characters(escaped_once)

print(f"Original: {s!r}")
print(f"Escaped once: {escaped_once!r}")
print(f"Escaped twice: {escaped_twice!r}")
```

Output:
```
Original: ':'
Escaped once: '__D'
Escaped twice: '__dunderD'
```

## Why This Is A Bug

The function escapes special characters in type declarations. The replacement rules include both `':'` → `'__D'` and `'__'` → `'__dunder'`. When applied to an already-escaped string containing `'__D'`, the `'__'` prefix gets re-escaped to `'__dunder'`, producing `'__dunderD'`.

While the current caching in `type_identifier_from_declaration` prevents this from happening in normal usage, the lack of idempotence violates a fundamental property of escape functions and could cause bugs if escaped strings are accidentally processed twice.

## Fix

The function should not re-escape already-escaped sequences. One approach is to use a non-ambiguous escape sequence that can't be confused with the input. For example, using delimiters or a different escape pattern that doesn't start with '__'.

Alternatively, the escape function could check if a string has already been escaped and return it unchanged. However, this is more complex and may not be practical.

A simpler fix is to ensure that the escape sequences themselves don't contain patterns that would be escaped again. For instance, using single underscores or a different marker that isn't in the replacement table.