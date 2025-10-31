# Bug Report: Cython.Compiler.Tests.TestTypes Incorrect Length Assertion for Internal Function

**Target**: `Cython.Compiler.Tests.TestTypes.TestTypeIdentifiers._test_escape`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The test `TestTypeIdentifiers._test_escape` incorrectly asserts that the internal function `_escape_special_type_characters` produces output ≤64 characters, but this function only escapes special characters and does not cap length by design.

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, Verbosity
import Cython.Compiler.PyrexTypes as PT

@given(st.text())
@settings(max_examples=1000, verbosity=Verbosity.verbose)
def test_escape_type_length_invariant(s):
    result = PT._escape_special_type_characters(s)
    assert len(result) <= 64, f"Length {len(result)} exceeds 64 for input: {repr(s)}"

if __name__ == "__main__":
    # Run the test
    test_escape_type_length_invariant()
```

<details>

<summary>
<b>Failing input</b>: <code>'000000000000000000000 0 0 0 0 '</code>
</summary>

```
Trying example: test_escape_type_length_invariant(
    s='000000000000000000000 0 0 0 0 ',
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 12, in test_escape_type_length_invariant
    assert len(result) <= 64, f"Length {len(result)} exceeds 64 for input: {repr(s)}"
           ^^^^^^^^^^^^^^^^^
AssertionError: Length 65 exceeds 64 for input: '000000000000000000000 0 0 0 0 '

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 16, in <module>
    test_escape_type_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 9, in test_escape_type_length_invariant
    @settings(max_examples=1000, verbosity=Verbosity.verbose)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 12, in test_escape_type_length_invariant
    assert len(result) <= 64, f"Length {len(result)} exceeds 64 for input: {repr(s)}"
           ^^^^^^^^^^^^^^^^^
AssertionError: Length 65 exceeds 64 for input: '000000000000000000000 0 0 0 0 '
Falsifying example: test_escape_type_length_invariant(
    s='000000000000000000000 0 0 0 0 ',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import _escape_special_type_characters

failing_input = '000000000000000000:<<,,,,'
result = _escape_special_type_characters(failing_input)

print(f"Input: '{failing_input}'")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")

# The test in TestTypes.py:75 expects this to be <= 64
try:
    assert len(result) <= 64, f"Length {len(result)} exceeds 64"
    print("Test PASSED: Length is within 64 characters")
except AssertionError as e:
    print(f"Test FAILED: {e}")
    print("\nThis demonstrates that _escape_special_type_characters can produce")
    print("output longer than 64 characters, violating the test's assertion.")
```

<details>

<summary>
AssertionError: Length 65 exceeds 64
</summary>

```
Input: '000000000000000000:<<,,,,'
Output: '000000000000000000__D__lAng__lAng__comma___comma___comma___comma_'
Output length: 65
Test FAILED: Length 65 exceeds 64

This demonstrates that _escape_special_type_characters can produce
output longer than 64 characters, violating the test's assertion.
```
</details>

## Why This Is A Bug

The test at `Cython/Compiler/Tests/TestTypes.py:75` incorrectly assumes that `_escape_special_type_characters` guarantees a 64-character limit, but this violates the function's design. The function `_escape_special_type_characters` is an internal utility (underscore prefix) whose sole purpose is to escape special characters in type declarations - it replaces characters like `:`, `<`, `,` with escaped versions like `__D`, `__lAng`, `__comma_`. It has no responsibility for length capping.

The public API `type_identifier_from_declaration` correctly handles length limits by:
1. First calling `_escape_special_type_characters` to escape special characters
2. Then calling `cap_length` to ensure the result is ≤64 characters

The test's `_test_escape` method applies the same length assertion to both functions, which is incorrect since only the public API guarantees the length constraint.

## Relevant Context

The code structure in `PyrexTypes.py` clearly shows the separation of concerns:

- Line 5699: `safe = _escape_special_type_characters(safe)` - escapes characters
- Line 5700: `safe = cap_length(re.sub(...))` - caps length after escaping
- Lines 5704-5708: `cap_length` function truncates strings > 63 chars with a hash prefix

The test passes with the existing TEST_DATA (lines 22-42) by coincidence - all test strings happen to produce escaped output ≤64 characters. However, the test fails for legitimate inputs like `'000000000000000000:<<,,,,'` which escapes to 65+ characters.

Documentation: The function is internal (underscore prefix) and undocumented, following Python convention that such functions are not part of the public API.

## Proposed Fix

```diff
--- a/Cython/Compiler/Tests/TestTypes.py
+++ b/Cython/Compiler/Tests/TestTypes.py
@@ -44,7 +44,8 @@ class TestTypeIdentifiers(unittest.TestCase):
     def test_escape_special_type_characters(self):
         test_func = PT._escape_special_type_characters  # keep test usage visible for IDEs
         function_name = "_escape_special_type_characters"
-        self._test_escape(function_name)
+        # Don't test length for internal function that doesn't guarantee it
+        self._test_escape_without_length_check(function_name)

     def test_type_identifier_for_declaration(self):
         test_func = PT.type_identifier_from_declaration  # keep test usage visible for IDEs
@@ -74,3 +75,11 @@ class TestTypeIdentifiers(unittest.TestCase):
                 func_name, declaration, escaped_value, expected))
             # test that the length has been successfully capped
             self.assertLessEqual(len(escaped_value), 64)
+
+    def _test_escape_without_length_check(self, func_name, test_data=TEST_DATA):
+        escape = getattr(PT, func_name)
+        for declaration, expected in test_data:
+            escaped_value = escape(declaration)
+            self.assertEqual(escaped_value, expected, "%s('%s') == '%s' != '%s'" % (
+                func_name, declaration, escaped_value, expected))
+            # Note: No length assertion for internal _escape_special_type_characters
```