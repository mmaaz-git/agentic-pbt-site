# Bug Report: pandas.io.clipboard.waitForNewPaste Returns Non-String Values

**Target**: `pandas.io.clipboard.waitForNewPaste`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `waitForNewPaste()` function violates its documented contract by returning non-string values (None, integers, booleans, lists, dictionaries) when the clipboard changes to these types, instead of waiting for an actual text string as promised in the docstring.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import pandas.io.clipboard as clipboard

@given(
    initial=st.text(),
    changed=st.one_of(st.none(), st.integers(), st.booleans(), st.lists(st.text(), max_size=3), st.dictionaries(st.text(), st.text(), max_size=3))
)
@example(initial="text", changed=None)  # Specific example that fails
@settings(max_examples=100, deadline=None)
def test_waitForNewPaste_should_wait_for_string(initial, changed):
    """Test that waitForNewPaste should only return strings, not other types"""

    # Skip if the initial and changed are the same (no change detected)
    if initial == changed:
        return

    # Save original paste function
    original_paste = clipboard.paste
    call_count = [0]

    def mock_paste():
        call_count[0] += 1
        return initial if call_count[0] == 1 else changed

    clipboard.paste = mock_paste

    try:
        result = clipboard.waitForNewPaste(timeout=0.1)

        # The function docstring says it waits for "a new text string"
        # and "returns this text" - so it should always return a string
        assert isinstance(result, str), \
            f"waitForNewPaste should return string, not {type(result).__name__}. " \
            f"Returned: {repr(result)}"

    finally:
        clipboard.paste = original_paste

if __name__ == "__main__":
    # Run the test
    test_waitForNewPaste_should_wait_for_string()
```

<details>

<summary>
**Failing input**: `initial='text', changed=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 41, in <module>
    test_waitForNewPaste_should_wait_for_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 5, in test_waitForNewPaste_should_wait_for_string
    initial=st.text(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 32, in test_waitForNewPaste_should_wait_for_string
    assert isinstance(result, str), \
           ~~~~~~~~~~^^^^^^^^^^^^^
AssertionError: waitForNewPaste should return string, not NoneType. Returned: None
Falsifying explicit example: test_waitForNewPaste_should_wait_for_string(
    initial='text',
    changed=None,
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard

# Save the original paste function
original_paste = clipboard.paste

# Create a mock counter to track calls
call_count = [0]

def mock_paste():
    """Mock paste function that returns 'initial text' on first call, None on second"""
    call_count[0] += 1
    if call_count[0] == 1:
        return "initial text"
    else:
        return None  # This should cause the bug

# Replace the paste function with our mock
clipboard.paste = mock_paste

# Call waitForNewPaste - it should wait for a new text string
# But instead it will return None when it sees the change
result = clipboard.waitForNewPaste(timeout=0.1)

print(f"Result: {repr(result)}")
print(f"Type: {type(result).__name__}")

# Restore original paste function
clipboard.paste = original_paste
```

<details>

<summary>
Returns None instead of waiting for text
</summary>
```
Result: None
Type: NoneType
```
</details>

## Why This Is A Bug

The function's docstring at line 715-721 explicitly states: "This function call blocks until a new text string exists on the clipboard that is different from the text that was there when the function was first called. It returns this text."

Key violations:
1. The documentation promises to wait for "a new **text string**" - emphasis on string type
2. The documentation states "It returns this **text**" - again emphasizing text/string return type
3. The current implementation at line 726 only checks `if currentText != originalText:` without verifying that `currentText` is actually a string

This bug can occur in real-world scenarios:
- On macOS with pyobjc, the `paste_osx_pyobjc()` function (line 127-131) can return None when `board.stringForType_(AppKit.NSStringPboardType)` returns None if the clipboard contains non-text data
- When clipboard contains images, files, or other non-text data types
- During clipboard transitions or when clipboard is cleared

## Relevant Context

The module already has type enforcement elsewhere:
- The `_stringifyText()` helper function (line 89-96) enforces that only `(str, int, float, bool)` can be copied, showing the module is designed for text operations
- The sister function `waitForPaste()` (line 694-711) has a similar issue, checking only `if clipboardText != "":` without type verification

Related code locations:
- Function definition: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/clipboard/__init__.py:714-733`
- Similar function with same issue: `waitForPaste()` at line 694-711
- Type checking helper: `_stringifyText()` at line 89-96

## Proposed Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -723,7 +723,7 @@ def waitForNewPaste(timeout=None):
     originalText = paste()
     while True:
         currentText = paste()
-        if currentText != originalText:
+        if isinstance(currentText, str) and currentText != originalText:
             return currentText
         time.sleep(0.01)
```