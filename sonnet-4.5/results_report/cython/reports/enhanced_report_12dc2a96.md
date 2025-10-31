# Bug Report: Cython.Debugger CyBreak.complete Duplicate Completion Suggestions

**Target**: `Cython.Debugger.libcython.CyBreak.complete`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CyBreak.complete` method incorrectly suggests already-typed function names in tab completion when the `word` parameter is empty, due to Python's slice notation `text[:-0]` returning an empty string instead of the original text.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test demonstrating the Cython debugger CyBreak.complete bug.
This test shows that already-typed function names appear in completion suggestions.
"""

from hypothesis import given, strategies as st, settings, example


def complete_unqualified_logic(text, word, all_names):
    """
    Extracted logic from CyBreak.complete method for unqualified name completion.
    This is the actual code from lines 957-959 of libcython.py.
    """
    word = word or ""
    seen = set(text[:-len(word)].split())
    return [n for n in all_names if n.startswith(word) and n not in seen]


@given(st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
@example("spam")  # Explicit example from the bug report
@settings(max_examples=10)
def test_complete_with_empty_word(funcname):
    """
    Test that demonstrates the bug: when word is empty, already-typed
    function names incorrectly appear in completion suggestions.
    """
    word = ""
    text = f"cy break {funcname} "  # User typed the function name and a space
    all_names = [funcname, "other_func", "another_func"]

    result = complete_unqualified_logic(text, word, all_names)

    # This assertion PASSES, confirming the bug exists
    # The function name should NOT be in the result since it's already typed
    assert funcname in result, f"Bug confirmed: '{funcname}' should not be in suggestions but is present"

    print(f"✓ Bug reproduced with funcname='{funcname}'")
    print(f"  text = {repr(text)}")
    print(f"  word = {repr(word)}")
    print(f"  result = {result}")
    print(f"  '{funcname}' incorrectly appears in completion suggestions")


if __name__ == "__main__":
    print("Running hypothesis test for CyBreak.complete bug...")
    print("=" * 60)
    test_complete_with_empty_word()
    print("=" * 60)
    print("All tests passed, confirming the bug exists.")
    print("The bug: already-typed function names appear in completions when word=''")
```

<details>

<summary>
**Failing input**: `funcname = "spam"`
</summary>
```
Running hypothesis test for CyBreak.complete bug...
============================================================
✓ Bug reproduced with funcname='spam'
  text = 'cy break spam '
  word = ''
  result = ['spam', 'other_func', 'another_func']
  'spam' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='A'
  text = 'cy break A '
  word = ''
  result = ['A', 'other_func', 'another_func']
  'A' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='O'
  text = 'cy break O '
  word = ''
  result = ['O', 'other_func', 'another_func']
  'O' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='ꭵäŅęRĩİÑĚl𝞤𐕽'
  text = 'cy break ꭵäŅęRĩİÑĚl𝞤𐕽 '
  word = ''
  result = ['ꭵäŅęRĩİÑĚl𝞤𐕽', 'other_func', 'another_func']
  'ꭵäŅęRĩİÑĚl𝞤𐕽' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='Õŋù𝚃č'
  text = 'cy break Õŋù𝚃č '
  word = ''
  result = ['Õŋù𝚃č', 'other_func', 'another_func']
  'Õŋù𝚃č' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='ĶŮ𐲂ſ'
  text = 'cy break ĶŮ𐲂ſ '
  word = ''
  result = ['ĶŮ𐲂ſ', 'other_func', 'another_func']
  'ĶŮ𐲂ſ' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='ħ'
  text = 'cy break ħ '
  word = ''
  result = ['ħ', 'other_func', 'another_func']
  'ħ' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='аīⱓŚdċŜ𝞷ŧ𝛑'
  text = 'cy break аīⱓŚdċŜ𝞷ŧ𝛑 '
  word = ''
  result = ['аīⱓŚdċŜ𝞷ŧ𝛑', 'other_func', 'another_func']
  'аīⱓŚdċŜ𝞷ŧ𝛑' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='NÆv'
  text = 'cy break NÆv '
  word = ''
  result = ['NÆv', 'other_func', 'another_func']
  'NÆv' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='xᎣ𐐅èâқὁ'
  text = 'cy break xᎣ𐐅èâқὁ '
  word = ''
  result = ['xᎣ𐐅èâқὁ', 'other_func', 'another_func']
  'xᎣ𐐅èâқὁ' incorrectly appears in completion suggestions
✓ Bug reproduced with funcname='ßⲲÓlļÇê'
  text = 'cy break ßⲲÓlļÇê '
  word = ''
  result = ['ßⲲÓlļÇê', 'other_func', 'another_func']
  'ßⲲÓlļÇê' incorrectly appears in completion suggestions
============================================================
All tests passed, confirming the bug exists.
The bug: already-typed function names appear in completions when word=''
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the Cython debugger CyBreak.complete bug.
This demonstrates the issue with text[:-0] returning an empty string.
"""

text = "cy break spam "
word = ""

print("Input:")
print(f"  text = {repr(text)}")
print(f"  word = {repr(word)}")
print()

# The bug: when word is empty, text[:-len(word)] becomes text[:-0]
# which in Python returns an empty string, not the original string
slice_result = text[:-len(word)]
print("Slicing operation:")
print(f"  len(word) = {len(word)}")
print(f"  text[:-len(word)] = text[:-{len(word)}] = {repr(slice_result)}")
print()

# This causes seen to be empty when it should contain the already-typed function names
seen = set(slice_result.split())
print("Result of splitting:")
print(f"  seen = set(text[:-len(word)].split()) = {seen}")
print()

# Simulate the completion logic
all_names = ["spam", "eggs", "ham", "bacon"]
result = [n for n in all_names if n.startswith(word) and n not in seen]
print("Completion result:")
print(f"  all_names = {all_names}")
print(f"  result = [n for n in all_names if n.startswith(word) and n not in seen]")
print(f"  result = {result}")
print()

print("Bug manifestation:")
print(f"  'spam' in result = {'spam' in result}")
print(f"  Expected: False (spam should be filtered out as it's already typed)")
print(f"  Actual: True (spam appears in suggestions due to empty seen set)")
```

<details>

<summary>
Duplicate completion suggestion for already-typed function name
</summary>
```
Input:
  text = 'cy break spam '
  word = ''

Slicing operation:
  len(word) = 0
  text[:-len(word)] = text[:-0] = ''

Result of splitting:
  seen = set(text[:-len(word)].split()) = set()

Completion result:
  all_names = ['spam', 'eggs', 'ham', 'bacon']
  result = [n for n in all_names if n.startswith(word) and n not in seen]
  result = ['spam', 'eggs', 'ham', 'bacon']

Bug manifestation:
  'spam' in result = True
  Expected: False (spam should be filtered out as it's already typed)
  Actual: True (spam appears in suggestions due to empty seen set)
```
</details>

## Why This Is A Bug

This violates expected tab completion behavior in several ways:

1. **Python Slice Notation Quirk**: The code at line 957 of `libcython.py` uses `text[:-len(word)]` to extract the already-typed portion of the command. However, when `word` is empty (length 0), this becomes `text[:-0]` which in Python returns an empty string `''` rather than the original text. This is a well-documented Python behavior where negative zero in slicing has special semantics.

2. **Intent vs Implementation Mismatch**: The variable name `seen` and the filtering logic `n not in seen` clearly indicate the developer's intent to track and filter out already-typed function names. The code comment at line 940 references the GDB source code for completion behavior, and standard GDB completion practice is to not suggest already-typed values.

3. **User Experience Impact**: When a user types `cy break spam ` (with trailing space) and presses TAB for completion, they expect to see new function name suggestions, not "spam" again. Tab completion systems universally filter out already-completed words to provide meaningful suggestions for what comes next.

4. **Contradicts GDB Completion Standards**: The GDB Python API documentation states that the `complete(text, word)` method should return completion strings. The standard behavior in GDB and similar debuggers is to suggest new options, not repeat what's already typed.

## Relevant Context

The bug occurs in the `CyBreak` class which implements the `cy break` command for the Cython debugger extension to GDB. This command allows setting breakpoints in Cython code using either qualified names (e.g., `module.Class.method`) or unqualified names (e.g., `function_name`).

The `complete` method is called by GDB when the user presses TAB for command completion. According to the GDB Python API:
- `text`: Contains the complete command line up to the cursor
- `word`: Contains the last word being completed (can be empty at word boundaries)

The bug specifically affects the "unqualified name" completion path (lines 956-959) when completing at a word boundary where `word` is empty. The code is located at:
`/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Debugger/libcython.py:957`

Link to similar GDB completion implementation referenced in the code:
https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-cmd.c;h=7143c1c5f7fdce9316a8c41fc2246bc6a07630d4;hb=HEAD#l140

## Proposed Fix

```diff
--- a/Cython/Debugger/libcython.py
+++ b/Cython/Debugger/libcython.py
@@ -954,7 +954,10 @@ class CyBreak(CythonCommand):
         words = text.strip().split()
         if not words or '.' not in words[-1]:
             # complete unqualified
-            seen = set(text[:-len(word)].split())
+            if len(word) == 0:
+                seen = set(text.split())
+            else:
+                seen = set(text[:-len(word)].split())
             return [n for n in all_names
                           if n.startswith(word) and n not in seen]
```