# Bug Report: pandas.io.clipboard.waitForPaste Returns Non-String Values Instead of Waiting

**Target**: `pandas.io.clipboard.waitForPaste`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `waitForPaste()` function violates its documented contract by immediately returning non-string values (None, 0, False, [], {}) instead of waiting for an actual non-empty text string to appear on the clipboard.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for pandas.io.clipboard.waitForPaste using Hypothesis"""

from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@given(value=st.one_of(st.none(), st.just(0), st.just(False), st.just([]), st.just({})))
@settings(max_examples=50, deadline=None)
def test_waitForPaste_should_wait_for_string(value):
    """Test that waitForPaste waits for non-empty string, not any non-empty-string value"""
    original_paste = clipboard.paste
    clipboard.paste = lambda: value

    try:
        result = clipboard.waitForPaste(timeout=0.1)
        assert isinstance(result, str) and result != "", \
            f"waitForPaste should wait for non-empty string, not {type(value).__name__}"
    except clipboard.PyperclipTimeoutException:
        # This is expected behavior - it should timeout waiting for a string
        pass
    finally:
        clipboard.paste = original_paste

if __name__ == "__main__":
    test_waitForPaste_should_wait_for_string()
    print("Test completed.")
```

<details>

<summary>
**Failing input**: `None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 25, in <module>
    test_waitForPaste_should_wait_for_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 8, in test_waitForPaste_should_wait_for_string
    @settings(max_examples=50, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 16, in test_waitForPaste_should_wait_for_string
    assert isinstance(result, str) and result != "", \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: waitForPaste should wait for non-empty string, not NoneType
Falsifying example: test_waitForPaste_should_wait_for_string(
    value=None,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstration of pandas.io.clipboard.waitForPaste bug with non-string values"""

import pandas.io.clipboard as clipboard

# Save original paste function
original_paste = clipboard.paste

# Test case 1: None value
print("Test Case 1: None value")
clipboard.paste = lambda: None
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 2: Integer 0
print("\nTest Case 2: Integer 0")
clipboard.paste = lambda: 0
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 3: Boolean False
print("\nTest Case 3: Boolean False")
clipboard.paste = lambda: False
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 4: Empty list
print("\nTest Case 4: Empty list []")
clipboard.paste = lambda: []
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 5: Empty dict
print("\nTest Case 5: Empty dict {}")
clipboard.paste = lambda: {}
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Restore original paste function
clipboard.paste = original_paste

print("\nExpected behavior: All tests should timeout since no non-empty string exists.")
print("Actual behavior: Functions return non-string values immediately.")
```

<details>

<summary>
waitForPaste returns non-string values immediately instead of waiting or timing out
</summary>
```
Test Case 1: None value
Result: None, Type: NoneType

Test Case 2: Integer 0
Result: 0, Type: int

Test Case 3: Boolean False
Result: False, Type: bool

Test Case 4: Empty list []
Result: [], Type: list

Test Case 5: Empty dict {}
Result: {}, Type: dict

Expected behavior: All tests should timeout since no non-empty string exists.
Actual behavior: Functions return non-string values immediately.
```
</details>

## Why This Is A Bug

The function's docstring explicitly states: "This function call blocks until a non-empty text string exists on the clipboard. It returns this text." However, the current implementation violates this documented contract in multiple ways:

1. **Type Safety Violation**: The function returns non-string types (None, int, bool, list, dict) when it should only return strings according to its documentation.

2. **Incorrect Waiting Logic**: The implementation uses `clipboardText != ""` (line 704) which evaluates to True for any value that's not exactly an empty string. In Python, this means:
   - `None != ""` → True (returns None immediately)
   - `0 != ""` → True (returns 0 immediately)
   - `False != ""` → True (returns False immediately)
   - `[] != ""` → True (returns empty list immediately)
   - `{} != ""` → True (returns empty dict immediately)

3. **Real-World Impact on macOS**: The `paste_osx_pyobjc()` function (used on macOS with pyobjc backend) calls `board.stringForType_(AppKit.NSStringPboardType)` which returns `None` when the clipboard doesn't contain string data. This means macOS users will get `None` returned immediately instead of the function waiting for actual text.

4. **Contract Violation**: The function promises to "block until a non-empty text string exists" but instead returns immediately with non-text values, breaking any code that relies on receiving a string.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/clipboard/__init__.py` at lines 694-711. The pandas clipboard module is based on the pyperclip library but includes its own implementation.

Key observations:
- The function has clear documentation stating it waits for and returns "text" (implying string type)
- The paste backends can legitimately return `None` in production scenarios (especially on macOS)
- The bug affects type safety and can cause downstream `AttributeError` or `TypeError` exceptions when code expects string methods
- This is not an edge case but a realistic scenario when clipboard contains non-text data or is empty on certain platforms

Documentation reference: The function docstring at line 695-700 clearly establishes the contract that this implementation violates.

## Proposed Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -701,7 +701,7 @@ def waitForPaste(timeout=None):
     startTime = time.time()
     while True:
         clipboardText = paste()
-        if clipboardText != "":
+        if isinstance(clipboardText, str) and clipboardText != "":
             return clipboardText
         time.sleep(0.01)
```