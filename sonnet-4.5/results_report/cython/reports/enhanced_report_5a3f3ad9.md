# Bug Report: Cython.Plex.Scanner Infinite Loop with Nullable Patterns

**Target**: `Cython.Plex.Scanner`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Scanner.read() enters an infinite loop when the Lexicon contains nullable patterns (patterns that can match empty strings) like `Rep()`, `Opt()`, or `Str('')`. The scanner repeatedly returns empty tokens at the same position instead of advancing through the input or raising an error.

## Property-Based Test

```python
from io import StringIO
from hypothesis import given, settings, strategies as st, example
from Cython.Plex import Lexicon, Range, Rep, Scanner


@settings(max_examples=10)
@given(st.text(min_size=1, max_size=20))
@example('abc')  # Add explicit example that we know fails
def test_scanner_should_terminate(text):
    """Test that Scanner.read() always terminates, even with nullable patterns."""
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

    assert len(tokens) < max_tokens, f"Scanner appears to be in infinite loop. Got {len(tokens)} tokens from input '{text}'"


if __name__ == "__main__":
    # Run the test with verbose output
    import sys
    from hypothesis import reproduce_failure

    print("Running Hypothesis test for Cython.Plex.Scanner infinite loop bug...")
    print("=" * 60)

    try:
        test_scanner_should_terminate()
        print("All tests passed!")
    except Exception as e:
        print(f"FAILED")
        print(f"Error: {e}")
        print("\nThis test demonstrates that Scanner.read() enters an infinite loop")
        print("when the Lexicon contains nullable patterns like Rep() that can match")
        print("empty strings. The scanner keeps returning empty tokens instead of")
        print("advancing through the input or raising an error.")
```

<details>

<summary>
**Failing input**: `text='abc'`
</summary>
```
Running Hypothesis test for Cython.Plex.Scanner infinite loop bug...
============================================================
FAILED
Error: Scanner appears to be in infinite loop. Got 130 tokens from input 'abc'

This test demonstrates that Scanner.read() enters an infinite loop
when the Lexicon contains nullable patterns like Rep() that can match
empty strings. The scanner keeps returning empty tokens instead of
advancing through the input or raising an error.
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from Cython.Plex import Lexicon, Range, Rep, Scanner

# Create a lexer that accepts zero or more digits
digit = Range('0', '9')
lexicon = Lexicon([(Rep(digit), 'INT')])

# Give it input that contains non-digit characters
scanner = Scanner(lexicon, StringIO('abc'), 'test')

# Try to read tokens - this will loop infinitely
print("Attempting to read tokens from 'abc' with Rep(digit) pattern:")
for i in range(10):
    token_type, token_text = scanner.read()
    print(f'Iteration {i}: token_type={token_type!r}, token_text={token_text!r}')
    if token_type is None:
        print("Reached end of file")
        break
else:
    print("Stopped after 10 iterations to prevent infinite loop")
```

<details>

<summary>
Scanner returns infinite empty tokens instead of advancing or raising error
</summary>
```
Attempting to read tokens from 'abc' with Rep(digit) pattern:
Iteration 0: token_type='INT', token_text=''
Iteration 1: token_type='INT', token_text=''
Iteration 2: token_type='INT', token_text=''
Iteration 3: token_type='INT', token_text=''
Iteration 4: token_type='INT', token_text=''
Iteration 5: token_type='INT', token_text=''
Iteration 6: token_type='INT', token_text=''
Iteration 7: token_type='INT', token_text=''
Iteration 8: token_type='INT', token_text=''
Iteration 9: token_type='INT', token_text=''
Stopped after 10 iterations to prevent infinite loop
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of lexical scanners in multiple ways:

1. **Scanner.read() documentation states** it should return `(None, '')` on end of file, but instead it loops infinitely returning empty tokens when nullable patterns are present.

2. **Rep() is a core regex operation** documented in the Plex API (`Rep(re)` matches zero or more repetitions). A lexer that cannot handle this fundamental pattern without entering an infinite loop is severely broken.

3. **Inconsistent behavior**: Rep1() (one or more) correctly raises `UnrecognizedInput` when it cannot match the input, demonstrating that Rep() should also handle non-matching input properly rather than looping infinitely.

4. **The bug occurs in multiple scenarios**:
   - When input contains non-matching characters (e.g., 'abc' with digit pattern)
   - After successfully matching valid input (e.g., after matching '123', continues returning empty tokens at EOF)
   - With any nullable pattern like `Opt()`, `Rep()`, or `Str('')`

5. **Resource exhaustion**: In production code, this causes programs to hang indefinitely and consume unbounded memory as tokens accumulate.

## Relevant Context

The bug is located in the `scan_a_token` method in `/Cython/Plex/Scanners.py` at lines 162-167. When `cur_pos == start_pos` (no characters consumed), the code only handles EOL and EOF cases, but fails to handle the case where a nullable pattern matches an empty string at a regular character position.

`Rep(re)` is implemented as `Opt(Rep1(re))` (see Regexps.py:494), meaning it can match empty string, which is the root cause of this issue.

Documentation: https://github.com/cython/cython/tree/master/Cython/Plex

## Proposed Fix

The scanner should detect when it makes an empty match at a non-EOF position and advance the input position by at least one character to ensure forward progress:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -160,11 +160,17 @@ class Scanner:
             return (text, action)
         else:
             if self.cur_pos == self.start_pos:
                 if self.cur_char is EOL:
                     self.next_char()
                 if self.cur_char is None or self.cur_char is EOF:
                     return ('', None)
+                # If we matched empty string but not at EOF, we need to advance
+                # to avoid infinite loop with nullable patterns like Rep() or Opt()
+                if action is not None:
+                    # We had a zero-width match, advance by one character
+                    self.next_char()
+                    return self.scan_a_token()
             raise Errors.UnrecognizedInput(self, self.state_name)
```