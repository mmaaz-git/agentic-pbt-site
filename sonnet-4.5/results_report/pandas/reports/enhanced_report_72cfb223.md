# Bug Report: Cython.Plex.Scanner Infinite Loop with Nullable Patterns

**Target**: `Cython.Plex.Scanner`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Scanner.read() enters an infinite loop when the Lexicon contains nullable patterns (patterns that can match empty strings) like `Rep()` or `Opt()`, repeatedly returning empty tokens without advancing the input position or raising an error.

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


if __name__ == "__main__":
    test_scanner_should_terminate()
```

<details>

<summary>
**Failing input**: `text='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 26, in <module>
    test_scanner_should_terminate()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 7, in test_scanner_should_terminate
    @given(st.text(min_size=1, max_size=20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 22, in test_scanner_should_terminate
    assert len(tokens) < max_tokens, f"Scanner in infinite loop"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Scanner in infinite loop
Falsifying example: test_scanner_should_terminate(
    text='0',
)
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from Cython.Plex import Lexicon, Range, Rep, Scanner

# Create a lexicon with a nullable pattern (Rep can match zero occurrences)
digit = Range('0', '9')
lexicon = Lexicon([(Rep(digit), 'INT')])

# Create scanner with input that cannot match digits
scanner = Scanner(lexicon, StringIO('abc'), 'test')

# Try to read tokens - this will loop infinitely
for i in range(10):
    token_type, token_text = scanner.read()
    print(f'{i}: token_type={token_type!r}, token_text={token_text!r}')
    if token_type is None:
        print("EOF reached")
        break

print("\nDemonstrating the bug after successful match:")
scanner2 = Scanner(lexicon, StringIO('123abc'), 'test')
for i in range(5):
    token_type, token_text = scanner2.read()
    print(f'{i}: token_type={token_type!r}, token_text={token_text!r}')
    if token_type is None:
        print("EOF reached")
        break
```

<details>

<summary>
Scanner enters infinite loop returning empty 'INT' tokens
</summary>
```
0: token_type='INT', token_text=''
1: token_type='INT', token_text=''
2: token_type='INT', token_text=''
3: token_type='INT', token_text=''
4: token_type='INT', token_text=''
5: token_type='INT', token_text=''
6: token_type='INT', token_text=''
7: token_type='INT', token_text=''
8: token_type='INT', token_text=''
9: token_type='INT', token_text=''

Demonstrating the bug after successful match:
0: token_type='INT', token_text='123'
1: token_type='INT', token_text=''
2: token_type='INT', token_text=''
3: token_type='INT', token_text=''
4: token_type='INT', token_text=''
```
</details>

## Why This Is A Bug

The Scanner violates expected lexer behavior by entering an infinite loop when patterns can match empty strings. According to the documentation:

1. **`Rep(re)` documentation states**: "Rep(re) is an RE which matches zero or more repetitions of |re|." The "zero or more" explicitly allows empty matches.

2. **`Opt(re)` documentation states**: "Opt(re) is an RE which matches either |re| or the empty string." This explicitly matches empty strings.

3. **Scanner.read() documentation states**: It should "Read the next lexical token from the stream" and "Returns (None, '') on end of file." There is no documented behavior for infinite empty token generation.

4. **Standard lexer behavior**: All standard lexer implementations (flex, Python's re module, etc.) handle zero-width matches by either advancing the input position by at least one character or raising an error when no progress can be made.

5. **Inconsistent with Rep1()**: The `Rep1()` function (which matches one or more repetitions) correctly raises `UnrecognizedInput` when it cannot match the input, demonstrating that proper error handling exists for similar patterns.

The bug occurs in the `scan_a_token()` method at line 153-160 of Scanners.py: when a pattern matches zero characters (`cur_pos == start_pos` after matching), and the current character is not EOL or EOF, it should either advance the position or raise an error, but instead it returns the empty match, causing the scanner to get stuck in the same position indefinitely.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Scanners.py`. The problematic code section is in the `scan_a_token()` method:

```python
def scan_a_token(self):
    """
    Read the next input sequence recognised by the machine
    and return (text, action). Returns ('', None) on end of
    file.
    """
    self.start_pos = self.cur_pos
    self.current_scanner_position_tuple = (
        self.name, self.cur_line, self.cur_pos - self.cur_line_start
    )
    action = self.run_machine_inlined()
    if action is not None:
        if self.trace:
            print("Scanner: read: Performing %s %d:%d" % (
                action, self.start_pos, self.cur_pos))
        text = self.buffer[
            self.start_pos - self.buf_start_pos:
            self.cur_pos - self.buf_start_pos]
        return (text, action)
    else:
        if self.cur_pos == self.start_pos:
            if self.cur_char is EOL:
                self.next_char()
            if self.cur_char is None or self.cur_char is EOF:
                return ('', None)
        raise Errors.UnrecognizedInput(self, self.state_name)
```

The issue is that when `action` is not None but the match consumed zero characters (empty match from Rep or Opt), the method returns the empty text without checking if progress was made. This causes the scanner to remain at the same position and repeat the same empty match infinitely.

## Proposed Fix

The scanner needs to detect zero-width matches and ensure forward progress. Here's a proposed fix:

```diff
--- a/Scanners.py
+++ b/Scanners.py
@@ -151,11 +151,17 @@ class Scanner:
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
+            if self.cur_pos == self.start_pos and self.cur_char not in (None, EOF):
+                # Force advance by one character to prevent infinite loop
+                self.next_char()
+                if self.cur_char is None or self.cur_char is EOF:
+                    return ('', None)
+                raise Errors.UnrecognizedInput(self, self.state_name)
             return (text, action)
         else:
```

Alternatively, the scanner could be modified to skip zero-width matches and continue scanning, or to treat them as an error condition immediately. The key is ensuring that the scanner always makes forward progress through the input stream.