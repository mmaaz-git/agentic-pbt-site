# Bug Report: pandas.util.capitalize_first_letter Unicode Length Change

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function changes string length and fails to preserve the suffix for certain Unicode characters that have multi-character uppercase forms when converted.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, Verbosity, assume, example
import pandas.util

@given(st.text(min_size=1))
@example('ß')  # German sharp s
@example('ßeta')  # German sharp s with suffix
@example('ﬁ')  # ligature fi
@example('ﬂ')  # ligature fl
@example('ﬆ')  # ligature st
@example('ﬁle')  # ligature fi with suffix
@settings(verbosity=Verbosity.verbose, max_examples=100)
def test_capitalize_first_letter_length_preservation(s):
    """Test that capitalize_first_letter preserves string length."""
    result = pandas.util.capitalize_first_letter(s)

    # Check if length is preserved
    if len(result) != len(s):
        print(f"\nFailing input: '{s}'")
        print(f"Input length: {len(s)}, Output length: {len(result)}")
        print(f"Input: '{s}' -> Output: '{result}'")
        print(f"Input chars: {[c for c in s]}")
        print(f"Output chars: {[c for c in result]}")

    assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"

# Run the test
if __name__ == "__main__":
    test_capitalize_first_letter_length_preservation()
```

<details>

<summary>
**Failing input**: `'ß'` (and 5 other examples)
</summary>
```

Failing input: 'ß'
Input length: 1, Output length: 2
Input: 'ß' -> Output: 'SS'
Input chars: ['ß']
Output chars: ['S', 'S']

Failing input: 'ßeta'
Input length: 4, Output length: 5
Input: 'ßeta' -> Output: 'SSeta'
Input chars: ['ß', 'e', 't', 'a']
Output chars: ['S', 'S', 'e', 't', 'a']

Failing input: 'ﬁ'
Input length: 1, Output length: 2
Input: 'ﬁ' -> Output: 'FI'
Input chars: ['ﬁ']
Output chars: ['F', 'I']

Failing input: 'ﬂ'
Input length: 1, Output length: 2
Input: 'ﬂ' -> Output: 'FL'
Input chars: ['ﬂ']
Output chars: ['F', 'L']

Failing input: 'ﬆ'
Input length: 1, Output length: 2
Input: 'ﬆ' -> Output: 'ST'
Input chars: ['ﬆ']
Output chars: ['S', 'T']

Failing input: 'ﬁle'
Input length: 3, Output length: 4
Input: 'ﬁle' -> Output: 'FIle'
Input chars: ['ﬁ', 'l', 'e']
Output chars: ['F', 'I', 'l', 'e']
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 28, in <module>
  |     test_capitalize_first_letter_length_preservation()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_capitalize_first_letter_length_preservation
  |     @example('ß')  # German sharp s
  |                   ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 6 distinct failures in explicit examples. (6 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ß' (len=1) -> 'SS' (len=2)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ß',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ßeta' (len=4) -> 'SSeta' (len=5)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ßeta',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ﬁ' (len=1) -> 'FI' (len=2)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ﬁ',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ﬂ' (len=1) -> 'FL' (len=2)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ﬂ',
    | )
    +---------------- 5 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ﬆ' (len=1) -> 'ST' (len=2)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ﬆ',
    | )
    +---------------- 6 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 24, in test_capitalize_first_letter_length_preservation
    |     assert len(result) == len(s), f"Length not preserved: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"
    |            ^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Length not preserved: 'ﬁle' (len=3) -> 'FIle' (len=4)
    | Falsifying explicit example: test_capitalize_first_letter_length_preservation(
    |     s='ﬁle',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.util

# Test case 1: German sharp s (ß) changes length from 1 to 2
input_str = 'ß'
result = pandas.util.capitalize_first_letter(input_str)
print(f"Input: '{input_str}' (length {len(input_str)})")
print(f"Output: '{result}' (length {len(result)})")
print(f"Expected: First letter capitalized, length preserved")
print(f"Actual: Length changed from {len(input_str)} to {len(result)}")
print()

# Test case 2: Suffix not preserved with 'ßeta'
input_str = 'ßeta'
result = pandas.util.capitalize_first_letter(input_str)
print(f"Input: '{input_str}'")
print(f"Output: '{result}'")
print(f"Input suffix (s[1:]): '{input_str[1:]}'")
print(f"Output suffix (result[1:]): '{result[1:]}'")
print(f"Expected: Suffix should be 'eta'")
print(f"Actual: Suffix is '{result[1:]}' instead of '{input_str[1:]}'")
print()

# Test case 3: Ligature fi (ﬁ) changes length from 3 to 4
input_str = 'ﬁle'
result = pandas.util.capitalize_first_letter(input_str)
print(f"Input: '{input_str}' (length {len(input_str)})")
print(f"Output: '{result}' (length {len(result)})")
print(f"Expected: First letter capitalized, length preserved")
print(f"Actual: Length changed from {len(input_str)} to {len(result)}")
print()

# Test case 4: Other ligatures
test_cases = [
    ('ﬂow', 'ligature fl'),
    ('ﬆart', 'ligature st'),
]

for input_str, description in test_cases:
    try:
        result = pandas.util.capitalize_first_letter(input_str)
        print(f"{description}: '{input_str}' -> '{result}' (length {len(input_str)} -> {len(result)})")
    except Exception as e:
        print(f"{description}: Error - {e}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Input: 'ß' (length 1)
Output: 'SS' (length 2)
Expected: First letter capitalized, length preserved
Actual: Length changed from 1 to 2

Input: 'ßeta'
Output: 'SSeta'
Input suffix (s[1:]): 'eta'
Output suffix (result[1:]): 'Seta'
Expected: Suffix should be 'eta'
Actual: Suffix is 'Seta' instead of 'eta'

Input: 'ﬁle' (length 3)
Output: 'FIle' (length 4)
Expected: First letter capitalized, length preserved
Actual: Length changed from 3 to 4

ligature fl: 'ﬂow' -> 'FLow' (length 3 -> 4)
ligature st: 'ﬆart' -> 'STart' (length 4 -> 5)
```
</details>

## Why This Is A Bug

The function `capitalize_first_letter` is implemented in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/__init__.py:28-29` as:

```python
def capitalize_first_letter(s):
    return s[:1].upper() + s[1:]
```

This implementation violates two properties strongly implied by the function's name:

1. **Length preservation violation**: The function name "capitalize_first_letter" implies it should only capitalize the first letter without changing the string's length. However, for certain Unicode characters that expand to multiple characters when uppercased (ß→SS, ﬁ→FI, ﬂ→FL, ﬆ→ST), the string length increases.

2. **Suffix modification**: When the first character expands to multiple characters, the suffix `s[1:]` is not properly preserved. For example, with input 'ßeta':
   - The suffix should be 'eta' (characters from position 1 onward)
   - But the result 'SSeta' has suffix 'Seta' (the second 'S' from the expanded 'SS' plus 'eta')
   - This means the original suffix content is altered, not just the first character

3. **Semantic mismatch**: The function name suggests it "capitalizes" the first letter, but it actually applies `.upper()` which is a different operation. Python's `.capitalize()` method handles these Unicode cases differently (e.g., 'ß'.capitalize() → 'Ss').

## Relevant Context

This bug affects Unicode characters that have special case-mapping rules defined in the Unicode standard:
- **German eszett (ß)**: Uppercases to 'SS' per German orthography rules
- **Typographic ligatures**: ﬁ→FI, ﬂ→FL, ﬆ→ST, etc.
- These are valid according to Unicode case mapping standards, but unexpected for a function named "capitalize_first_letter"

The function is located in the pandas.util module but is not documented in the official pandas API documentation, suggesting it may be an internal utility function. However, it is still accessible via `pandas.util.capitalize_first_letter` and could be used by other parts of the codebase or by users directly.

## Proposed Fix

Since the function's current behavior follows correct Unicode standards but violates expectations based on its name, here are two possible fixes:

**Option 1: Preserve length and handle Unicode edge cases**
```diff
 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    if not s:
+        return s
+    first_upper = s[0].upper()
+    # If uppercasing expands to multiple characters, use capitalize() instead
+    if len(first_upper) > 1:
+        return s[0] + s[1:]  # Keep original first char if it would expand
+    return first_upper + s[1:]
```

**Option 2: Document the current behavior**
```diff
 def capitalize_first_letter(s):
+    """
+    Uppercase the first character of a string and append the rest unchanged.
+
+    Note: For some Unicode characters (e.g., 'ß'), the uppercase form
+    may be multiple characters ('SS'), which will change the string length
+    and modify the apparent suffix.
+
+    Parameters
+    ----------
+    s : str
+        Input string
+
+    Returns
+    -------
+    str
+        String with first character uppercased
+    """
     return s[:1].upper() + s[1:]
```