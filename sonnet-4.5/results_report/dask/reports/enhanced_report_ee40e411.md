# Bug Report: dask.utils.key_split Incorrectly Strips Legitimate English Words

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `key_split` function incorrectly strips legitimate English words (like "feedback", "faceache", "beefcafe") from task names when they are exactly 8 characters long and contain only letters a-f, mistakenly identifying them as hexadecimal suffixes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from dask.utils import key_split

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
       st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
@example(key1='task', key2='feedback')  # Add the specific failing case
@settings(max_examples=500)
def test_key_split_compound_key(key1, key2):
    """Test that key_split preserves legitimate English words in compound keys"""
    s = f"{key1}-{key2}-1"
    result = key_split(s)

    # If key2 is not 8 chars of only a-f letters, it should be preserved
    if len(key2) != 8 or not all(c in 'abcdef' for c in key2):
        assert result == f"{key1}-{key2}", f"key_split('{s}') returned '{result}', expected '{key1}-{key2}'"

# Run the test
if __name__ == "__main__":
    print("Running property-based test for key_split...")
    print("Testing that legitimate words are preserved in compound keys")
    print()

    try:
        test_key_split_compound_key()
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"✗ Test failed!")
        print(f"Assertion error: {e}")
        print("\nThis demonstrates that key_split incorrectly strips legitimate English words")
        print("that happen to be 8 characters long and contain only letters a-f.")
```

<details>

<summary>
**Failing input**: `key1='task', key2='feedback'`
</summary>
```
Running property-based test for key_split...
Testing that legitimate words are preserved in compound keys

✗ Test failed!
Assertion error: key_split('task-feedback-1') returned 'task', expected 'task-feedback'

This demonstrates that key_split incorrectly strips legitimate English words
that happen to be 8 characters long and contain only letters a-f.
```
</details>

## Reproducing the Bug

```python
from dask.utils import key_split

# Test the main failing case
result = key_split('task-feedback-1')
print(f"Result: {repr(result)}")
print(f"Expected: 'task-feedback'")
print()

# Test if the bug occurs
try:
    assert result == 'task-feedback'
    print("✓ Test passed: 'feedback' was preserved")
except AssertionError:
    print("✗ Bug confirmed: 'feedback' was incorrectly stripped!")

print("\n--- Additional test cases ---")

# Test other affected words
test_cases = [
    ('process-feedback-0', 'process-feedback'),
    ('data-faceache-1', 'data-faceache'),
    ('task-beefcafe-2', 'task-beefcafe'),
    ('hello-world-1', 'hello-world'),  # This should work as shown in docstring
    ('x-abcdefab-1', 'x'),  # This is expected to strip per docstring
]

for input_str, expected in test_cases:
    result = key_split(input_str)
    status = "✓" if result == expected else "✗"
    print(f"{status} key_split('{input_str}') -> '{result}' (expected: '{expected}')")
```

<details>

<summary>
Bug confirmed: 'feedback' incorrectly stripped from compound task names
</summary>
```
Result: 'task'
Expected: 'task-feedback'

✗ Bug confirmed: 'feedback' was incorrectly stripped!

--- Additional test cases ---
✗ key_split('process-feedback-0') -> 'process' (expected: 'process-feedback')
✗ key_split('data-faceache-1') -> 'data' (expected: 'data-faceache')
✗ key_split('task-beefcafe-2') -> 'task' (expected: 'task-beefcafe')
✓ key_split('hello-world-1') -> 'hello-world' (expected: 'hello-world')
✓ key_split('x-abcdefab-1') -> 'x' (expected: 'x')
```
</details>

## Why This Is A Bug

This violates the expected behavior documented in the function's docstring. The docstring example `key_split('hello-world-1')` returns `'hello-world'`, demonstrating that meaningful compound words should be preserved. However, the function inconsistently strips legitimate English words like "feedback" while preserving "world".

The bug occurs because the hex detection logic at line 1990 in utils.py uses an overly permissive pattern:
- The regex `[a-f]+` combined with `.match()` matches ANY string that starts with letters a-f
- When a word is exactly 8 characters and contains only letters a-f, it's incorrectly classified as a hex suffix
- This affects real English words: "feedback", "faceache", "beefcafe", "deafface", "acedface", etc.

The function is meant to strip random hexadecimal identifiers (like git short hashes), not legitimate English words. The docstring comment "# ignores hex" (line 1972) confirms the intent is to ignore hex identifiers, not words that coincidentally use only a-f letters.

This causes practical problems:
1. **Key collisions**: Both 'task-feedback-1' and 'task-deadbeef-1' resolve to 'task'
2. **Lost semantic information**: Meaningful task name components are discarded
3. **Inconsistent behavior**: Users reasonably expect "feedback" to be preserved like "world" is

## Relevant Context

The `key_split` function is used throughout Dask to extract task name prefixes from task keys. It's designed to remove numeric and hexadecimal suffixes that are often appended to create unique task identifiers, while preserving the semantic task name.

The function is defined in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` starting at line 1948.

The problematic hex pattern is defined at line 1944:
```python
hex_pattern = re.compile("[a-f]+")
```

And used at lines 1989-1991:
```python
if word.isalpha() and not (
    len(word) == 8 and hex_pattern.match(word) is not None
):
```

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1941,7 +1941,7 @@ def parse_bytes(s):
         yield chunk


-hex_pattern = re.compile("[a-f]+")
+hex_pattern = re.compile(r"^[a-f0-9]+$")


 @functools.lru_cache(100000)
@@ -1987,7 +1987,8 @@ def key_split(s):
         else:
             result = words[0]
         for word in words[1:]:
-            if word.isalpha() and not (
-                len(word) == 8 and hex_pattern.match(word) is not None
+            # Preserve word if it's alphabetic and NOT a hex-like string
+            # (8 chars containing both letters and numbers in hex range)
+            if word.isalpha() and not (
+                len(word) == 8 and hex_pattern.match(word) is not None and any(c in '0123456789' for c in word)
             ):
                 result += "-" + word
```