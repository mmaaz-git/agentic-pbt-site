# Bug Report: Cython.Plex Eof Action Value Ignored

**Target**: `Cython.Plex.Scanners.Scanner` with `Eof` pattern
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an `Eof` pattern is specified in a Lexicon with an associated action value, the Scanner ignores the action and always returns `None` at end-of-file, violating the documented Lexicon behavior.

## Property-Based Test

```python
import io
from hypothesis import given, strategies as st, settings
from Cython.Plex import *

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=200)
def test_eof_action_arbitrary_value(value):
    lexicon = Lexicon([
        (Str('x'), 'X'),
        (Eof, value)
    ])

    scanner = Scanner(lexicon, io.StringIO('x'))
    scanner.read()

    token_eof, text_eof = scanner.read()
    assert token_eof == value, f"Eof should return its action value {value!r}, got {token_eof!r}"
```

**Failing input**: `value=1` (or any value - Eof always returns `None`)

## Reproducing the Bug

```python
import io
from Cython.Plex import *

lexicon = Lexicon([
    (Str('x'), 'X'),
    (Eof, 'EOF_TOKEN')
])

scanner = Scanner(lexicon, io.StringIO('x'))

token1, text1 = scanner.read()
print(f"First token: {token1!r}")

token2, text2 = scanner.read()
print(f"EOF token: {token2!r}")
print(f"Expected: 'EOF_TOKEN', Actual: {token2!r}")
```

Output:
```
First token: 'X'
EOF token: None
Expected: 'EOF_TOKEN', Actual: None
```

## Why This Is A Bug

The Lexicon specification states that actions determine the token value returned. For all normal patterns, this works correctly:

```python
lexicon = Lexicon([(Str('hello'), 'HELLO_TOKEN')])
scanner = Scanner(lexicon, io.StringIO('hello'))
token, text = scanner.read()
# token == 'HELLO_TOKEN' ✓
```

However, the `Eof` pattern completely ignores its action:

```python
lexicon = Lexicon([(Eof, 'EOF_TOKEN')])
scanner = Scanner(lexicon, io.StringIO(''))
token, text = scanner.read()
# token == None ✗ (expected 'EOF_TOKEN')
```

This behavior is inconsistent with the documented Lexicon specification and prevents:
1. Detecting EOF with a custom token value
2. Using Eof with callbacks or special handling
3. Distinguishing between "no match" and "matched EOF"

All other patterns (Str, Any, Range, etc.) correctly return their action values, making Eof's behavior an anomaly that violates the pattern-action contract.

## Fix

The Scanner's EOF handling logic needs to be modified to respect the Eof action value instead of hardcoding `None`. This likely requires changes in the `read()` method where EOF is detected and the return value is constructed.