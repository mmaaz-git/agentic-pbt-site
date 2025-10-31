# Bug Report: Cython.Tempita Template._repr Incorrect Exception Constructor Arguments

**Target**: `Cython.Tempita._tempita.Template._repr`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Template._repr` method incorrectly constructs `UnicodeDecodeError` and `UnicodeEncodeError` exceptions with a single string argument instead of the required 5 arguments, causing a confusing TypeError when these specific error paths are triggered.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from Cython.Tempita import Template

class SpecialBytes:
    """A class that triggers UnicodeDecodeError in __str__"""
    def __init__(self, data):
        self.data = data

    def __str__(self):
        # This raises UnicodeDecodeError when str() is called on it
        raise UnicodeDecodeError('utf-8', self.data, 0, 1, 'invalid start byte')

    def __bytes__(self):
        return self.data

@given(st.binary(min_size=1, max_size=10))
@example(b'\xff')  # Explicit example that should fail
@settings(max_examples=100)
def test_template_repr_unicode_error_construction(byte_value):
    """
    Test that Template._repr properly constructs UnicodeDecodeError.

    When a bytes-like object raises UnicodeDecodeError in its __str__ method
    and the template has no default_encoding, the code should raise a proper
    UnicodeDecodeError. However, due to a bug, it raises TypeError instead.
    """
    content = "{{x}}"
    template = Template(content)
    template.default_encoding = None  # Force the buggy code path

    special_obj = SpecialBytes(byte_value)

    try:
        result = template.substitute({'x': special_obj})
        # Should not reach here - should raise an error
        assert False, "Expected an exception but none was raised"
    except TypeError as e:
        # This is the bug - we get TypeError instead of UnicodeDecodeError
        # The TypeError message should be about wrong number of arguments
        assert "takes exactly 5 arguments" in str(e) or "required positional argument" in str(e)
        print(f"Bug confirmed: Got TypeError '{e}' instead of UnicodeDecodeError")
    except UnicodeDecodeError as e:
        # This would be the correct behavior (after fix)
        assert "Cannot decode bytes value" in str(e)
        print(f"Correct behavior: Got UnicodeDecodeError: {e}")

if __name__ == "__main__":
    # Run the test function
    test_template_repr_unicode_error_construction()
    print("\nHypothesis test completed - bug reproduced successfully")
```

<details>

<summary>
**Failing input**: `b'\xff'` (or any bytes value with the special SpecialBytes wrapper)
</summary>
```
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError
Bug confirmed: Got TypeError 'function takes exactly 5 arguments (1 given)' instead of UnicodeDecodeError

Hypothesis test completed - bug reproduced successfully
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Create a special class that triggers the bug
class SpecialBytes:
    def __str__(self):
        # This raises UnicodeDecodeError when str() is called on it
        raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

    def __bytes__(self):
        return b'\xff'

# Create a template with a placeholder
content = "{{x}}"
template = Template(content)

# Set default_encoding to None to trigger the buggy code path
template.default_encoding = None

# Try to substitute with our special object
try:
    result = template.substitute({'x': SpecialBytes()})
    print("No error raised - unexpected!")
except TypeError as e:
    print(f"TypeError raised (this is the bug): {e}")
    print(f"\nExpected: UnicodeDecodeError with message about 'Cannot decode bytes value...'")
    print(f"Actual: TypeError about UnicodeDecodeError constructor needing 5 arguments")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError raised correctly: {e}")
```

<details>

<summary>
TypeError about incorrect constructor arguments
</summary>
```
TypeError raised (this is the bug): function takes exactly 5 arguments (1 given)

Expected: UnicodeDecodeError with message about 'Cannot decode bytes value...'
Actual: TypeError about UnicodeDecodeError constructor needing 5 arguments
```
</details>

## Why This Is A Bug

The Python built-in exceptions `UnicodeDecodeError` and `UnicodeEncodeError` require exactly 5 arguments in their constructor: `(encoding, object, start, end, reason)`. However, in the Cython Tempita code at lines 353-355 and 367-369 of `_tempita.py`, these exceptions are incorrectly constructed with a single formatted string argument.

When the code path is triggered (which requires a custom object that raises UnicodeDecodeError in its `__str__` method and a template with `default_encoding` set to None), instead of getting the intended helpful error message like "Cannot decode bytes value b'\\xff' into unicode (no default_encoding provided)", users receive a confusing TypeError: "function takes exactly 5 arguments (1 given)".

This violates the expected behavior because:
1. The code clearly intends to raise a UnicodeDecodeError/UnicodeEncodeError with a descriptive message
2. The same file shows the correct pattern for constructing these exceptions at lines 359-364
3. The resulting TypeError masks the actual encoding issue and provides no useful information to the user

## Relevant Context

The bug exists in two locations in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py`:

1. **Lines 353-355**: Incorrect UnicodeDecodeError construction when trying to decode bytes without default_encoding
2. **Lines 367-369**: Incorrect UnicodeEncodeError construction when trying to encode unicode without default_encoding

The correct pattern for constructing these exceptions is demonstrated in the same file at lines 359-364, where a caught UnicodeDecodeError is properly re-raised with all 5 required arguments.

This bug requires a specific scenario to trigger:
- A custom object that raises UnicodeDecodeError in its `__str__()` method
- A Template instance with `default_encoding` set to None
- The template being unicode-based (content is a str, not bytes)

While this is an edge case, it's still a legitimate bug that produces confusing error messages when encountered.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -350,9 +350,9 @@ class Template:
         else:
             if self._unicode and isinstance(value, bytes):
                 if not self.default_encoding:
-                    raise UnicodeDecodeError(
+                    raise ValueError(
                         'Cannot decode bytes value %r into unicode '
                         '(no default_encoding provided)' % value)
                 try:
                     value = value.decode(self.default_encoding)
                 except UnicodeDecodeError as e:
@@ -364,9 +364,9 @@ class Template:
                         e.reason + ' in string %r' % value)
             elif not self._unicode and isinstance(value, str):
                 if not self.default_encoding:
-                    raise UnicodeEncodeError(
+                    raise ValueError(
                         'Cannot encode unicode value %r into bytes '
                         '(no default_encoding provided)' % value)
                 value = value.encode(self.default_encoding)
             return value
```