# Bug Report: numpy.f2py.symbolic.eliminate_quotes AssertionError on Unmatched Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `eliminate_quotes` function in NumPy's F2PY module crashes with an AssertionError when processing strings containing unmatched quote characters, and silently returns incorrect results when Python is run with optimization flags enabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_eliminate_insert_quotes_roundtrip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    restored = symbolic.insert_quotes(new_s, mapping)
    assert restored == s

# Run the test
if __name__ == "__main__":
    test_eliminate_insert_quotes_roundtrip()
```

<details>

<summary>
**Failing input**: `s='"'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 12, in <module>
    test_eliminate_insert_quotes_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 5, in test_eliminate_insert_quotes_roundtrip
    def test_eliminate_insert_quotes_roundtrip(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 6, in test_eliminate_insert_quotes_roundtrip
    new_s, mapping = symbolic.eliminate_quotes(s)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1194, in eliminate_quotes
    assert '"' not in new_s
           ^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_eliminate_insert_quotes_roundtrip(
    s='"',
)
```
</details>

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

# Test case that crashes with unmatched quote
print("Testing eliminate_quotes with single unmatched double quote:")
print("Input: '\"'")
try:
    result = symbolic.eliminate_quotes('"')
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting eliminate_quotes with single unmatched single quote:")
print("Input: \"'\"")
try:
    result = symbolic.eliminate_quotes("'")
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
```

<details>

<summary>
AssertionError raised for both unmatched quote types
</summary>
```
Testing eliminate_quotes with single unmatched double quote:
Input: '"'
AssertionError raised

Testing eliminate_quotes with single unmatched single quote:
Input: "'"
AssertionError raised
```
</details>

## Why This Is A Bug

The `eliminate_quotes` function uses assertions to validate its internal state, which is inappropriate for input validation. The function's regex patterns (lines 1188-1192) are designed to match complete quoted strings with both opening and closing quotes. When given an unmatched quote, the regex doesn't match it, leaving the quote character in the processed string `new_s`. This triggers the assertions at lines 1194-1195:

```python
assert '"' not in new_s
assert "'" not in new_s
```

This violates expected behavior in three ways:

1. **Inappropriate error type**: AssertionError is meant for internal invariants, not input validation. Users parsing malformed Fortran code receive a cryptic error instead of a meaningful exception like ValueError.

2. **Silent corruption with -O flag**: When Python runs with optimization enabled (`python -O`), assertions are disabled. The function then returns `('"', {})` for input `'"'`, silently producing incorrect output that violates the roundtrip property with `insert_quotes`.

3. **Violation of error handling best practices**: The function is part of F2PY's public API (used in `Parser.parse()` at line 1313) and should handle invalid input gracefully with proper exceptions, not assertions.

## Relevant Context

The `eliminate_quotes` function is part of F2PY's symbolic expression parser for Fortran code. It temporarily replaces quoted strings with placeholder tokens to simplify parsing, with `insert_quotes` reversing the operation. The function supports:
- Single and double quoted strings
- Kind-prefixed strings (e.g., `MYSTRKIND_"ABC"`)
- Escaped quotes within strings

The existing test suite (`test_symbolic.py`, lines 37-49) only tests valid, properly matched quotes. No test cases exist for error conditions or malformed input.

When run with `python -O` (optimization mode), the function produces incorrect output:
```python
# With python -O:
symbolic.eliminate_quotes('"')  # Returns: ('"', {})
symbolic.eliminate_quotes("'")  # Returns: ("'", {})
```

## Proposed Fix

Replace assertions with proper input validation that raises a semantic exception:

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1191,8 +1191,10 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s
+    if '"' in new_s or "'" in new_s:
+        raise ValueError(
+            f"Unmatched quotes found in input string: {s!r}"
+        )

     return new_s, d
```