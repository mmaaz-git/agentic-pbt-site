# Bug Report: Flask Environment Variable Parsers Don't Strip Whitespace

**Target**: `flask.helpers.get_debug_flag` and `flask.helpers.get_load_dotenv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Flask helper functions `get_debug_flag()` and `get_load_dotenv()` fail to strip whitespace from environment variable values before parsing, causing unexpected behavior when environment variables contain leading or trailing spaces.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for Flask environment variable whitespace bug.
Uses Hypothesis to systematically test environment variable parsing.
"""

from hypothesis import given, strategies as st, settings
from flask.helpers import get_debug_flag, get_load_dotenv
import os


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  ", " \t", "\n", "\r\n"])
)
@settings(max_examples=20, print_blob=True)
def test_debug_flag_whitespace_stripping(value, whitespace):
    """Test that get_debug_flag() handles whitespace consistently."""
    # Test with leading whitespace
    os.environ['FLASK_DEBUG'] = whitespace + value
    result_leading = get_debug_flag()

    # Test without whitespace
    os.environ['FLASK_DEBUG'] = value
    expected = get_debug_flag()

    # Test with trailing whitespace
    os.environ['FLASK_DEBUG'] = value + whitespace
    result_trailing = get_debug_flag()

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )

    assert result_trailing == expected, (
        f"Trailing whitespace changed result: {value + whitespace!r} → {result_trailing}, "
        f"expected {expected}"
    )


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  ", " \t", "\n", "\r\n"])
)
@settings(max_examples=20, print_blob=True)
def test_load_dotenv_whitespace_stripping(value, whitespace):
    """Test that get_load_dotenv() handles whitespace consistently."""
    # Test with leading whitespace
    os.environ['FLASK_SKIP_DOTENV'] = whitespace + value
    result_leading = get_load_dotenv(True)

    # Test without whitespace
    os.environ['FLASK_SKIP_DOTENV'] = value
    expected = get_load_dotenv(True)

    # Test with trailing whitespace
    os.environ['FLASK_SKIP_DOTENV'] = value + whitespace
    result_trailing = get_load_dotenv(True)

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )

    assert result_trailing == expected, (
        f"Trailing whitespace changed result: {value + whitespace!r} → {result_trailing}, "
        f"expected {expected}"
    )


if __name__ == "__main__":
    print("Running Hypothesis tests for Flask environment variable whitespace handling...")
    print("=" * 70)

    try:
        test_debug_flag_whitespace_stripping()
        print("✓ test_debug_flag_whitespace_stripping passed")
    except AssertionError as e:
        print(f"✗ test_debug_flag_whitespace_stripping failed: {e}")

    try:
        test_load_dotenv_whitespace_stripping()
        print("✓ test_load_dotenv_whitespace_stripping passed")
    except AssertionError as e:
        print(f"✗ test_load_dotenv_whitespace_stripping failed: {e}")
```

<details>

<summary>
**Failing input**: `value='false', whitespace=' '`
</summary>
```
Running Hypothesis tests for Flask environment variable whitespace handling...
======================================================================
✗ test_debug_flag_whitespace_stripping failed: Leading whitespace changed result: ' false' → True, expected False
✗ test_load_dotenv_whitespace_stripping failed: Leading whitespace changed result: ' false' → False, expected True
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Flask environment variable whitespace bug.
This demonstrates that Flask's get_debug_flag() and get_load_dotenv()
functions fail to strip whitespace from environment variable values.
"""

import os
from flask.helpers import get_debug_flag, get_load_dotenv


def test_debug_flag():
    """Test get_debug_flag() with and without whitespace."""
    print("Testing get_debug_flag():")
    print("-" * 40)

    # Test 1: Normal 'false' value (expected: False)
    os.environ['FLASK_DEBUG'] = 'false'
    result1 = get_debug_flag()
    print(f"FLASK_DEBUG='false' → {result1}")

    # Test 2: 'false' with leading space (expected: False, actual: True)
    os.environ['FLASK_DEBUG'] = ' false'
    result2 = get_debug_flag()
    print(f"FLASK_DEBUG=' false' → {result2}")

    # Test 3: 'false' with trailing space (expected: False, actual: True)
    os.environ['FLASK_DEBUG'] = 'false '
    result3 = get_debug_flag()
    print(f"FLASK_DEBUG='false ' → {result3}")

    # Test 4: 'false' with tab (expected: False, actual: True)
    os.environ['FLASK_DEBUG'] = '\tfalse'
    result4 = get_debug_flag()
    print(f"FLASK_DEBUG='\\tfalse' → {result4}")

    print(f"\nExpected all to be False")
    print(f"Actual results: {[result1, result2, result3, result4]}")

    # Demonstrate the problem
    assert result1 == False, "Normal 'false' should return False"
    try:
        assert result2 == False, "' false' should return False but returns True!"
    except AssertionError as e:
        print(f"\n❌ BUG FOUND: {e}")


def test_load_dotenv():
    """Test get_load_dotenv() with and without whitespace."""
    print("\n\nTesting get_load_dotenv():")
    print("-" * 40)

    # Test 1: Normal 'false' value (expected: True)
    os.environ['FLASK_SKIP_DOTENV'] = 'false'
    result1 = get_load_dotenv(True)
    print(f"FLASK_SKIP_DOTENV='false' → {result1}")

    # Test 2: 'false' with leading space (expected: True, actual: False)
    os.environ['FLASK_SKIP_DOTENV'] = ' false'
    result2 = get_load_dotenv(True)
    print(f"FLASK_SKIP_DOTENV=' false' → {result2}")

    # Test 3: 'false' with trailing space (expected: True, actual: False)
    os.environ['FLASK_SKIP_DOTENV'] = 'false '
    result3 = get_load_dotenv(True)
    print(f"FLASK_SKIP_DOTENV='false ' → {result3}")

    # Test 4: 'false' with tab (expected: True, actual: False)
    os.environ['FLASK_SKIP_DOTENV'] = '\tfalse'
    result4 = get_load_dotenv(True)
    print(f"FLASK_SKIP_DOTENV='\\tfalse' → {result4}")

    print(f"\nExpected all to be True")
    print(f"Actual results: {[result1, result2, result3, result4]}")

    # Demonstrate the problem
    assert result1 == True, "Normal 'false' should return True"
    try:
        assert result2 == True, "' false' should return True but returns False!"
    except AssertionError as e:
        print(f"\n❌ BUG FOUND: {e}")


if __name__ == "__main__":
    test_debug_flag()
    test_load_dotenv()

    print("\n" + "=" * 50)
    print("SUMMARY: Whitespace in environment variable values")
    print("causes incorrect parsing in Flask helper functions.")
    print("=" * 50)
```

<details>

<summary>
Output demonstrating the bug behavior
</summary>
```
Testing get_debug_flag():
----------------------------------------
FLASK_DEBUG='false' → False
FLASK_DEBUG=' false' → True
FLASK_DEBUG='false ' → True
FLASK_DEBUG='\tfalse' → True

Expected all to be False
Actual results: [False, True, True, True]

❌ BUG FOUND: ' false' should return False but returns True!


Testing get_load_dotenv():
----------------------------------------
FLASK_SKIP_DOTENV='false' → True
FLASK_SKIP_DOTENV=' false' → False
FLASK_SKIP_DOTENV='false ' → False
FLASK_SKIP_DOTENV='\tfalse' → False

Expected all to be True
Actual results: [True, False, False, False]

❌ BUG FOUND: ' false' should return True but returns False!

==================================================
SUMMARY: Whitespace in environment variable values
causes incorrect parsing in Flask helper functions.
==================================================
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Semantic Inconsistency**: The value `"false"` and `" false"` should be semantically equivalent when parsing configuration values. The whitespace does not change the intended meaning.

2. **Documentation Contradiction**: The Flask documentation for these functions does not specify that whitespace is significant. The docstrings state:
   - `get_debug_flag()`: "Get whether debug mode should be enabled for the app, indicated by the :envvar:`FLASK_DEBUG` environment variable"
   - `get_load_dotenv()`: "Get whether the user has disabled loading default dotenv files by setting :envvar:`FLASK_SKIP_DOTENV`"

   Neither mentions that whitespace affects parsing.

3. **Principle of Least Surprise**: Most configuration parsers (ConfigParser, YAML parsers, .env file parsers) strip whitespace by default. Users expect this behavior.

4. **Real-World Impact**: Environment variables commonly acquire whitespace through:
   - Shell script formatting with indentation
   - Manual exports with accidental spaces: `export FLASK_DEBUG="false "`
   - Configuration management tools that format values
   - `.env` files with trailing spaces

5. **Inverted Logic**: The bug causes counterintuitive behavior where `FLASK_DEBUG=" false"` **enables** debug mode (when it should disable it) and `FLASK_SKIP_DOTENV=" false"` **skips** dotenv loading (when it should load).

## Relevant Context

The bug originates from the implementation in `/home/npc/pbt/agentic-pbt/envs/flask_env/lib/python3.13/site-packages/flask/helpers.py`:

For `get_debug_flag()` (line 28-33):
```python
def get_debug_flag() -> bool:
    val = os.environ.get("FLASK_DEBUG")
    return bool(val and val.lower() not in {"0", "false", "no"})
```

For `get_load_dotenv()` (line 36-48):
```python
def get_load_dotenv(default: bool = True) -> bool:
    val = os.environ.get("FLASK_SKIP_DOTENV")
    if not val:
        return default
    return val.lower() in ("0", "false", "no")
```

Both functions call `val.lower()` directly without stripping whitespace first. When `val` is `" false"`, `val.lower()` becomes `" false"` which doesn't match the expected values `"false"`, `"0"`, or `"no"`.

Flask documentation: https://flask.palletsprojects.com/en/stable/api/#flask.helpers.get_debug_flag

## Proposed Fix

```diff
--- a/flask/helpers.py
+++ b/flask/helpers.py
@@ -30,6 +30,8 @@ def get_debug_flag() -> bool:
     :envvar:`FLASK_DEBUG` environment variable. The default is ``False``.
     """
     val = os.environ.get("FLASK_DEBUG")
+    if val:
+        val = val.strip()
     return bool(val and val.lower() not in {"0", "false", "no"})


@@ -44,6 +46,7 @@ def get_load_dotenv(default: bool = True) -> bool:

     if not val:
         return default
+    val = val.strip()
     return val.lower() in ("0", "false", "no")
```