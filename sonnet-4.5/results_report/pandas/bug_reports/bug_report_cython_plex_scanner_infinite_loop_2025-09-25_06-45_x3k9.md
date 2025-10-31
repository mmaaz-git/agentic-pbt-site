# Bug Report: Cython.Plex Scanner Infinite Loop with Nullable Patterns

**Target**: `Cython.Plex.Scanner`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Scanner.read() enters an infinite loop when the Lexicon contains nullable patterns (patterns that can match empty strings) like `Rep()` or `Opt()`. The scanner repeatedly returns empty tokens instead of advancing or raising an error.

## Property-Based Test

```python
from io import StringIO
from hypothesis import given, settings, strategies as st
from Cython.Plex import Lexicon, Range, Rep, Scanner


@settings(max_examples=10)
@given(st.text(min_size=1, max_size=20))
def test_scanner_should_terminate(text):
    digit = Range('0', '9')
    lexicon = Lexicon([(Rep(digit), 'INT')])
    scanner = Scanner(lexicon, StringIO(text), 'test')

    tokens = []
    max_tokens = len(text) * 10 + 100

    for _ in range(max_tokens):
        token_type, token_text = scanner.read()
        if token_type is None:
            break
        tokens.append((token_type, token_text))

    assert len(tokens) < max_tokens, f"Scanner in infinite loop"
```

**Failing input**: `text='abc'` (or any text with non-digit characters)

## Reproducing the Bug

```python
from io import StringIO
from Cython.Plex import Lexicon, Range, Rep, Scanner

digit = Range('0', '9')
lexicon = Lexicon([(Rep(digit), 'INT')])
scanner = Scanner(lexicon, StringIO('abc'), 'test')

for i in range(10):
    token_type, token_text = scanner.read()
    print(f'{i}: {token_type!r} = {token_text!r}')
```

Output shows infinite empty 'INT' tokens:
```
0: 'INT' = ''
1: 'INT' = ''
2: 'INT' = ''
...
```

The bug also occurs after successfully matching tokens:
```python
lexicon = Lexicon([(Rep(digit), 'INT')])
scanner = Scanner(lexicon, StringIO('123'), 'test')

for i in range(5):
    token_type, token_text = scanner.read()
    print(f'{i}: {token_type!r} = {token_text!r}')
```

Output:
```
0: 'INT' = '123'
1: 'INT' = ''
2: 'INT' = ''
...
```

## Why This Is A Bug

1. **Violates expected behavior**: Scanner.read() should either return matching tokens, raise UnrecognizedInput for non-matching input, or return (None, None) at EOF. It should never loop infinitely.

2. **Makes fundamental patterns unusable**: `Rep()` and `Opt()` are core regex operations documented in the API. A lexer that cannot handle these patterns is fundamentally broken.

3. **Inconsistent with Rep1**: `Rep1()` correctly raises UnrecognizedInput when given non-matching input, showing that Rep() should also handle this case properly rather than looping.

4. **Resource exhaustion**: In production code, this causes the program to hang and consume unbounded memory.

## Fix

The Scanner should detect when it makes an empty match and either:
1. Advance the input position by at least one character to prevent getting stuck
2. Raise an error if it cannot make progress

Most lexer implementations handle this by detecting zero-width matches and ensuring forward progress. The fix would likely be in `Scanners.py` in the `read()` or `scan_a_token()` methods to check if a match consumed zero characters and handle it appropriately.