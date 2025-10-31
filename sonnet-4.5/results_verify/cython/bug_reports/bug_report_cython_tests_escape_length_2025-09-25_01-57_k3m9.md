# Bug Report: Cython.Compiler.Tests.TestTypes Length Assertion Failure

**Target**: `Cython.Compiler.Tests.TestTypes.TestTypeIdentifiers._test_escape`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The test `TestTypeIdentifiers._test_escape` incorrectly asserts that `_escape_special_type_characters` produces output ≤64 characters, but this internal function has no such guarantee. Only the public-facing `type_identifier_from_declaration` maintains this invariant through subsequent `cap_length` calls.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import Cython.Compiler.PyrexTypes as PT

@given(st.text())
@settings(max_examples=1000)
def test_escape_type_length_invariant(s):
    result = PT._escape_special_type_characters(s)
    assert len(result) <= 64
```

**Failing input**: `'000000000000000000:<<,,,,'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, 'lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import _escape_special_type_characters

failing_input = '000000000000000000:<<,,,,'
result = _escape_special_type_characters(failing_input)

assert len(result) == 65
print(f"Output: {result}")
print(f"Length: {len(result)} > 64")
```

Output: `000000000000000000__D__lAng__lAng__comma___comma___comma___comma_`
Length: 65

## Why This Is A Bug

The test at `TestTypes.py:75` asserts `self.assertLessEqual(len(escaped_value), 64)` for the internal function `_escape_special_type_characters`, but this function doesn't guarantee length capping. The test passes with the existing TEST_DATA by coincidence (all examples happen to be short), but fails for other valid inputs.

The public API `type_identifier_from_declaration` correctly caps length by calling `cap_length()` after applying `_escape_special_type_characters`, so user-facing behavior is correct.

## Fix

```diff
--- a/Cython/Compiler/Tests/TestTypes.py
+++ b/Cython/Compiler/Tests/TestTypes.py
@@ -44,7 +44,6 @@ class TestTypeIdentifiers(unittest.TestCase):
     def test_escape_special_type_characters(self):
         test_func = PT._escape_special_type_characters  # keep test usage visible for IDEs
         function_name = "_escape_special_type_characters"
-        self._test_escape(function_name)
+        self._test_escape_without_length_check(function_name)

     def test_type_identifier_for_declaration(self):
@@ -68,6 +67,13 @@ class TestTypeIdentifiers(unittest.TestCase):
     def _test_escape(self, func_name, test_data=TEST_DATA):
         escape = getattr(PT, func_name)
         for declaration, expected in test_data:
             escaped_value = escape(declaration)
             self.assertEqual(escaped_value, expected, "%s('%s') == '%s' != '%s'" % (
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
```