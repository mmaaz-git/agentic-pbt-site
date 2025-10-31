# Bug Report: django.core.management.utils.handle_extensions Returns Invalid '.' Extension

**Target**: `django.core.management.utils.handle_extensions`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `handle_extensions` function incorrectly produces a single dot '.' as a file extension when processing comma-separated lists containing empty string segments, violating the function's purpose of returning valid file extensions.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test that discovers the handle_extensions bug producing invalid '.' extension.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet=' ,', min_size=1, max_size=20), min_size=1, max_size=5))
def test_handle_extensions_no_single_dot(separator_strings):
    result = handle_extensions(separator_strings)
    assert '.' not in result, f"Invalid extension '.' should not be in result, but got {result} from input {separator_strings}"

if __name__ == "__main__":
    # Run the test
    test_handle_extensions_no_single_dot()
```

<details>

<summary>
**Failing input**: `[',']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 18, in <module>
    test_handle_extensions_no_single_dot()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 12, in test_handle_extensions_no_single_dot
    def test_handle_extensions_no_single_dot(separator_strings):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 14, in test_handle_extensions_no_single_dot
    assert '.' not in result, f"Invalid extension '.' should not be in result, but got {result} from input {separator_strings}"
           ^^^^^^^^^^^^^^^^^
AssertionError: Invalid extension '.' should not be in result, but got {'.'} from input [',']
Falsifying example: test_handle_extensions_no_single_dot(
    separator_strings=[','],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the handle_extensions bug that produces invalid '.' extension.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.core.management.utils import handle_extensions

print("Test case 1: Empty string")
result = handle_extensions([''])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 2: Double comma")
result = handle_extensions(['html,,css'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 3: Trailing comma")
result = handle_extensions(['html,'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 4: Leading comma")
result = handle_extensions([',html'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 5: Just a comma")
result = handle_extensions([','])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("All test cases confirm the bug: handle_extensions produces invalid '.' extension")
```

<details>

<summary>
All test cases demonstrate '.' appearing in result sets
</summary>
```
Test case 1: Empty string
Result: {'.'}
✓ Confirmed: '.' is in the result

Test case 2: Double comma
Result: {'.', '.html', '.css'}
✓ Confirmed: '.' is in the result

Test case 3: Trailing comma
Result: {'.', '.html'}
✓ Confirmed: '.' is in the result

Test case 4: Leading comma
Result: {'.', '.html'}
✓ Confirmed: '.' is in the result

Test case 5: Just a comma
Result: {'.'}
✓ Confirmed: '.' is in the result

All test cases confirm the bug: handle_extensions produces invalid '.' extension
```
</details>

## Why This Is A Bug

This violates expected behavior because a single dot '.' is not a valid file extension. The function is documented to "organize multiple extensions that are separated with commas" and is used by Django's `makemessages` management command to match files with specific extensions.

The function's docstring examples only demonstrate valid extensions like '.html', '.js', and '.py', implying the function should return valid, usable file extensions. In file systems, '.' represents the current directory, not a file type identifier. When the makemessages command uses '.' as an extension filter, it would match no files (since no files have just a dot as their extension), causing silent failures where users don't understand why their files aren't being processed.

The bug occurs because the function splits comma-separated strings without filtering empty segments. When `'html,,css'.split(',')` produces `['html', '', 'css']`, the empty string gets prefixed with a dot, becoming '.'. This happens with common typos like double commas, trailing commas, or leading commas that users might accidentally enter when using the `--extension` flag.

## Relevant Context

The `handle_extensions` function is located at `/django/core/management/utils.py` lines 34-53. It's imported and used directly by the `makemessages` management command at line 373 of `/django/core/management/commands/makemessages.py`:

```python
self.extensions = handle_extensions(exts)
```

The makemessages command documentation states users should "Separate multiple extensions with commas" when using the `--extension` flag, making this a user-facing interface where typos are likely.

The function processes extensions by:
1. Splitting each input string by commas (line 49)
2. Removing spaces (line 49)
3. Adding a '.' prefix if not present (lines 51-52)
4. Returning a set of unique extensions (line 53)

The issue is at line 51-52 where empty strings pass the `not ext.startswith(".")` check and get '.' prepended, creating the invalid extension.

## Proposed Fix

```diff
--- a/django/core/management/utils.py
+++ b/django/core/management/utils.py
@@ -48,7 +48,9 @@ def handle_extensions(extensions):
     for ext in extensions:
         ext_list.extend(ext.replace(" ", "").split(","))
     for i, ext in enumerate(ext_list):
-        if not ext.startswith("."):
+        if not ext:
+            continue
+        elif not ext.startswith("."):
             ext_list[i] = ".%s" % ext_list[i]
-    return set(ext_list)
+    return {e for e in set(ext_list) if e and e != '.'}
```