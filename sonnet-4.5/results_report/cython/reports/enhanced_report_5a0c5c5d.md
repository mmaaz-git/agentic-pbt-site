# Bug Report: Cython.Debugger.DebugWriter.is_valid_tag Type-Dependent Inconsistency

**Target**: `Cython.Debugger.DebugWriter.is_valid_tag`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_valid_tag` function validates XML tag names differently depending on input type: it incorrectly accepts invalid names like `.0` when passed as regular strings but correctly rejects them when passed as EncodedString objects.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find inconsistencies in is_valid_tag.
This test checks that the function returns the same result regardless of
whether the input is a regular str or an EncodedString.
"""

from hypothesis import given, strategies as st, settings, example
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString


@given(st.text())
@example('.0')
@example('.123')
@example('.999')
@settings(max_examples=1000)
def test_is_valid_tag_consistency(s):
    """
    Test that is_valid_tag returns the same result for both str and EncodedString inputs.
    The function should behave consistently regardless of input type.
    """
    regular_result = is_valid_tag(s)
    encoded_result = is_valid_tag(EncodedString(s))
    assert regular_result == encoded_result, \
        f"Inconsistent: is_valid_tag({s!r}) = {regular_result}, but is_valid_tag(EncodedString({s!r})) = {encoded_result}"


if __name__ == "__main__":
    # Run the test
    test_is_valid_tag_consistency()
```

<details>

<summary>
**Failing input**: `.0` (and patterns `.123`, `.999`)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 31, in <module>
  |     test_is_valid_tag_consistency()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 14, in test_is_valid_tag_consistency
  |     @example('.0')
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 25, in test_is_valid_tag_consistency
    |     assert regular_result == encoded_result, \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent: is_valid_tag('.0') = True, but is_valid_tag(EncodedString('.0')) = False
    | Falsifying explicit example: test_is_valid_tag_consistency(
    |     s='.0',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 25, in test_is_valid_tag_consistency
    |     assert regular_result == encoded_result, \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent: is_valid_tag('.123') = True, but is_valid_tag(EncodedString('.123')) = False
    | Falsifying explicit example: test_is_valid_tag_consistency(
    |     s='.123',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 25, in test_is_valid_tag_consistency
    |     assert regular_result == encoded_result, \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent: is_valid_tag('.999') = True, but is_valid_tag(EncodedString('.999')) = False
    | Falsifying explicit example: test_is_valid_tag_consistency(
    |     s='.999',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for Cython.Debugger.DebugWriter.is_valid_tag bug.
This demonstrates the type-dependent behavior where the function returns different
results for the same logical input depending on whether it's a str or EncodedString.
"""

from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

# Test with the failing input ".0"
test_input = ".0"

# Test with regular string
result_str = is_valid_tag(test_input)
print(f'is_valid_tag("{test_input}") = {result_str}')

# Test with EncodedString
result_encoded = is_valid_tag(EncodedString(test_input))
print(f'is_valid_tag(EncodedString("{test_input}")) = {result_encoded}')

# These should be equal, but they are not
print(f"\nInconsistency detected: {result_str} != {result_encoded}")

# Test with a few more similar patterns
for test in [".123", ".999", ".1", ".00"]:
    str_result = is_valid_tag(test)
    enc_result = is_valid_tag(EncodedString(test))
    print(f'\nInput: "{test}"')
    print(f"  Regular string: {str_result}")
    print(f"  EncodedString:  {enc_result}")
    print(f"  Match: {str_result == enc_result}")

# This assertion will fail, demonstrating the bug
assert result_str == result_encoded, \
    f"Type-dependent behavior: is_valid_tag({test_input!r}) returns {result_str}, " \
    f"but is_valid_tag(EncodedString({test_input!r})) returns {result_encoded}"
```

<details>

<summary>
AssertionError: Type-dependent behavior
</summary>
```
is_valid_tag(".0") = True
is_valid_tag(EncodedString(".0")) = False

Inconsistency detected: True != False

Input: ".123"
  Regular string: True
  EncodedString:  False
  Match: False

Input: ".999"
  Regular string: True
  EncodedString:  False
  Match: False

Input: ".1"
  Regular string: True
  EncodedString:  False
  Match: False

Input: ".00"
  Regular string: True
  EncodedString:  False
  Match: False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 35, in <module>
    assert result_str == result_encoded, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Type-dependent behavior: is_valid_tag('.0') returns True, but is_valid_tag(EncodedString('.0')) returns False
```
</details>

## Why This Is A Bug

This violates the principle of type transparency and contradicts the function's documented purpose. The function's docstring explicitly states it should filter out "Names like '.0'" that are "used internally for arguments to functions creating generator expressions" because they "are not identifiers." The referenced GitHub issue #5552 confirms these names cause XML validation errors when generating debug information with lxml, which rejects '.0' as an "Invalid tag name."

The current implementation only performs this filtering when the input is an `EncodedString`, not when it's a regular `str`. This creates an API inconsistency where the same logical name ('.0') is treated differently based solely on its type wrapper. Since both `str` and `EncodedString` represent the same invalid XML tag name that would cause lxml to raise a `ValueError`, the function should consistently reject these names regardless of input type.

The function is used in three places within `CythonDebugWriter` (`start()`, `end()`, and `add_entry()` methods) specifically to prevent invalid XML tag names from being written to debug output. The type-dependent behavior means that if a regular string '.0' were passed to these methods, it would incorrectly be allowed through, potentially causing XML validation errors downstream.

## Relevant Context

The `is_valid_tag` function is located in `/Cython/Debugger/DebugWriter.py` at lines 16-27. The `EncodedString` class is a subclass of `str` defined in `/Cython/Compiler/StringEncoding.py` that tracks the original encoding of strings in the Cython compiler.

GitHub issue #5552 (https://github.com/cython/cython/issues/5552) provides the historical context: Cython's `--gdb` debug mode was failing when generator expressions had arguments with names like '.0', because these are not valid XML element names. Cython uses these names internally (similar to Python's internal iterator locals), but they must be filtered out from debug XML output.

The function correctly identifies and rejects these invalid names when passed as `EncodedString` objects (which is how they're typically represented in the Cython compiler), but fails to apply the same validation to regular strings.

## Proposed Fix

```diff
def is_valid_tag(name):
    """
    Names like '.0' are used internally for arguments
    to functions creating generator expressions,
    however they are not identifiers.

    See https://github.com/cython/cython/issues/5552
    """
-   if isinstance(name, EncodedString):
-       if name.startswith(".") and name[1:].isdecimal():
-           return False
+   if name.startswith(".") and len(name) > 1 and name[1:].isdecimal():
+       return False
    return True
```