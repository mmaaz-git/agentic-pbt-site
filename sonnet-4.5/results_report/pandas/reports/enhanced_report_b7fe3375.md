# Bug Report: Cython.Plex Scanner Infinite Loop with Nullable Patterns

**Target**: `Cython.Plex.Scanner`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Scanner.read() method enters an infinite loop when the Lexicon contains nullable patterns (patterns that can match empty strings) like `Rep()` or `Opt()`. When these patterns match zero characters, the scanner returns empty tokens indefinitely without advancing the input position or raising an error.

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

    assert len(tokens) < max_tokens, f"Scanner in infinite loop with input {text!r}"


if __name__ == "__main__":
    test_scanner_should_terminate()
```

<details>

<summary>
**Failing input**: `text='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 26, in <module>
    test_scanner_should_terminate()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 7, in test_scanner_should_terminate
    @given(st.text(min_size=1, max_size=20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 22, in test_scanner_should_terminate
    assert len(tokens) < max_tokens, f"Scanner in infinite loop with input {text!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Scanner in infinite loop with input '0'
Falsifying example: test_scanner_should_terminate(
    text='0',
)
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from Cython.Plex import Lexicon, Range, Rep, Scanner

# Create a lexicon with a nullable pattern (Rep can match zero characters)
digit = Range('0', '9')
lexicon = Lexicon([(Rep(digit), 'INT')])

# Test with non-digit characters - this will trigger the infinite loop
print("Testing with input 'abc':")
scanner = Scanner(lexicon, StringIO('abc'), 'test')

for i in range(10):
    token_type, token_text = scanner.read()
    print(f'{i}: {token_type!r} = {token_text!r}')
    if token_type is None:
        print("End of input")
        break

print("\n" + "=" * 50 + "\n")

# Test with digits first, then non-digits - also shows the problem
print("Testing with input '123abc':")
scanner2 = Scanner(lexicon, StringIO('123abc'), 'test')

for i in range(10):
    token_type, token_text = scanner2.read()
    print(f'{i}: {token_type!r} = {token_text!r}')
    if token_type is None:
        print("End of input")
        break
```

<details>

<summary>
Output shows infinite empty 'INT' tokens being returned
</summary>
```
Testing with input 'abc':
0: 'INT' = ''
1: 'INT' = ''
2: 'INT' = ''
3: 'INT' = ''
4: 'INT' = ''
5: 'INT' = ''
6: 'INT' = ''
7: 'INT' = ''
8: 'INT' = ''
9: 'INT' = ''

==================================================

Testing with input '123abc':
0: 'INT' = '123'
1: 'INT' = ''
2: 'INT' = ''
3: 'INT' = ''
4: 'INT' = ''
5: 'INT' = ''
6: 'INT' = ''
7: 'INT' = ''
8: 'INT' = ''
9: 'INT' = ''
```
</details>

## Why This Is A Bug

This violates expected scanner behavior in several critical ways:

1. **Infinite loop causes program hang**: The scanner gets stuck returning empty tokens forever, causing the program to hang and consume unbounded CPU and memory resources. This makes any code using nullable patterns completely unusable.

2. **Fundamental regex operations broken**: `Rep()` (zero or more) and `Opt()` (optional) are essential regex operations documented in the Cython.Plex API. These are not edge cases - they're core functionality that any lexer should handle properly.

3. **Inconsistent behavior**: The related pattern `Rep1()` (one or more) correctly raises `UnrecognizedInput` when it can't match, showing that the library knows how to handle non-matching input. The fact that `Rep()` loops infinitely instead is inconsistent.

4. **No forward progress guarantee**: Every lexer must ensure forward progress through the input. When a zero-width match occurs, the scanner should either advance by at least one character or raise an error, never get stuck at the same position.

5. **Silent failure mode**: The scanner doesn't raise an exception or give any indication of the problem - it just silently loops forever, making debugging difficult in production code.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Scanners.py` in the `scan_a_token()` method around lines 142-168. When a pattern successfully matches zero characters, the method returns the empty match without checking if any input was consumed. This causes the scanner to remain at the same position and match the same empty string again on the next call.

The issue affects any pattern that can match zero characters:
- `Rep(pattern)` - zero or more repetitions
- `Opt(pattern)` - optional pattern
- `Alt(Rep(pattern), other)` - alternatives containing nullable patterns

Documentation for these patterns: The Cython.Plex module documentation lists Rep and Opt as standard pattern constructors without any warnings about their use in lexicons.

## Proposed Fix

The scanner should detect zero-width matches and ensure forward progress. Here's a fix for the `scan_a_token()` method in Scanners.py:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -151,11 +151,19 @@ class Scanner:
         )
         action = self.run_machine_inlined()
         if action is not None:
             if self.trace:
                 print("Scanner: read: Performing %s %d:%d" % (
                     action, self.start_pos, self.cur_pos))
             text = self.buffer[
                 self.start_pos - self.buf_start_pos:
                 self.cur_pos - self.buf_start_pos]
+            # Check for zero-width match
+            if self.cur_pos == self.start_pos:
+                # We matched but consumed no characters - need to advance
+                # to prevent infinite loop
+                if self.cur_char is not EOL and self.cur_char is not EOF:
+                    # Skip one character to ensure progress
+                    self.next_char()
+                    raise Errors.UnrecognizedInput(self, self.state_name)
             return (text, action)
         else:
             if self.cur_pos == self.start_pos:
```

This fix detects when a pattern matches zero characters and either advances the input by one character or raises an appropriate error, preventing the infinite loop while maintaining correct scanner behavior.