# Bug Report: Cython.Plex.Regexps.RE.wrong_type AttributeError on Python 3

**Target**: `Cython.Plex.Regexps.RE.wrong_type`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RE.wrong_type` method crashes with an AttributeError when attempting to raise type validation errors because it uses `types.InstanceType` which was removed in Python 3, preventing proper error reporting for invalid RE arguments.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test that discovered the Cython.Plex.Regexps.RE.wrong_type bug
"""

import pytest
from hypothesis import given, strategies as st, settings
from Cython.Plex import Seq, Str
from Cython.Plex.Errors import PlexTypeError

@given(st.text(alphabet='abc', min_size=1, max_size=5))
@settings(max_examples=200)
def test_seq_rejects_non_re_args(s):
    """Test that Seq properly rejects non-RE arguments with PlexTypeError"""
    with pytest.raises(PlexTypeError):
        # The second argument to Seq should be an RE, not a string
        # This should raise PlexTypeError with a helpful message
        Seq(Str(s), "not an RE")

if __name__ == "__main__":
    # Run the test manually without Hypothesis decoration
    try:
        with pytest.raises(PlexTypeError):
            Seq(Str('a'), "not an RE")
        print("Test passed (PlexTypeError was raised as expected)")
    except AttributeError as e:
        print(f"Test failed with AttributeError: {e}")
        print("\nExpected: PlexTypeError with message about invalid type")
        print("Actual: AttributeError because types.InstanceType doesn't exist in Python 3")
        print("\nFailing input: s='a' (or any string value)")
```

<details>

<summary>
**Failing input**: `s='a'` (or any string value)
</summary>
```
Test failed with AttributeError: module 'types' has no attribute 'InstanceType'

Expected: PlexTypeError with message about invalid type
Actual: AttributeError because types.InstanceType doesn't exist in Python 3

Failing input: s='a' (or any string value)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for Cython.Plex.Regexps.RE.wrong_type bug
This demonstrates that wrong_type crashes with AttributeError in Python 3
instead of raising the intended PlexTypeError.
"""

from Cython.Plex import Seq, Str
from Cython.Plex.Errors import PlexTypeError

try:
    # This should raise PlexTypeError with a helpful message about wrong type
    # Instead it crashes with AttributeError about missing types.InstanceType
    seq = Seq(Str('a'), "not an RE")
except PlexTypeError as e:
    print(f"PlexTypeError (expected): {e}")
except AttributeError as e:
    print(f"AttributeError (unexpected): {e}")
    print("\nThis should have raised PlexTypeError with a helpful message.")
    print("Instead it crashes because types.InstanceType doesn't exist in Python 3.")
```

<details>

<summary>
AttributeError crash instead of PlexTypeError
</summary>
```
AttributeError (unexpected): module 'types' has no attribute 'InstanceType'

This should have raised PlexTypeError with a helpful message.
Instead it crashes because types.InstanceType doesn't exist in Python 3.
```
</details>

## Why This Is A Bug

This bug violates expected behavior in several critical ways:

1. **Broken Error Handling**: The `wrong_type` method (Regexps.py:166-174) is designed to raise `PlexTypeError` with a descriptive message when type validation fails. Instead, it crashes with `AttributeError` at line 167 because `types.InstanceType` was removed in Python 3.0.

2. **API Contract Violation**: The Plex module defines `PlexTypeError` in Errors.py specifically for type validation errors. When users pass invalid arguments to RE constructors like `Seq`, `Alt`, or `Rep1`, they should receive a `PlexTypeError` with a message like "Invalid type for argument 1 of Plex.Seq (expected Plex.RE instance, got str)". Instead, they get a confusing AttributeError about a missing Python 2 attribute.

3. **Python 3 Incompatibility**: The code uses `types.InstanceType` which only existed in Python 2 for old-style classes. This attribute was removed in Python 3.0 (released in 2008) as all classes became new-style classes. The code has never worked properly in Python 3.

4. **Validation Flow Disruption**: The validation chain `Seq.__init__` → `check_re` → `wrong_type` → `PlexTypeError` is broken at the `wrong_type` step, preventing proper error reporting for all RE type validations throughout the module.

## Relevant Context

The bug occurs in the type checking infrastructure used by all RE (Regular Expression) constructors in Cython's Plex module. The Plex module is used for lexical analysis and scanner generation within Cython.

Key code locations:
- Bug location: `/Cython/Plex/Regexps.py:167` - the problematic line using `types.InstanceType`
- Error class definition: `/Cython/Plex/Errors.py:12-13` - defines `PlexTypeError`
- Validation flow: `RE.check_re()` at line 151-153 calls `wrong_type()` when validation fails

The issue affects Python 3.x (tested on Python 3.13.2) and impacts all code that uses Plex RE constructors with invalid arguments. This makes debugging user errors significantly harder as the actual validation error is masked by the AttributeError.

Documentation: While Cython's main documentation doesn't detail Plex internals, the code clearly shows the intended error handling behavior through the defined exception hierarchy and method names.

## Proposed Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -164,7 +164,10 @@ class RE:
                                             num, self.__class__.__name__, repr(value)))

     def wrong_type(self, num, value, expected):
-        if type(value) == types.InstanceType:
+        # In Python 3, all objects have __class__, no need for InstanceType check
+        # Check if it's a class instance (not a built-in type)
+        if hasattr(value, '__class__') and hasattr(value.__class__, '__module__') and \
+           value.__class__.__module__ not in ('builtins', '__builtin__'):
             got = "%s.%s instance" % (
                 value.__class__.__module__, value.__class__.__name__)
         else:
```