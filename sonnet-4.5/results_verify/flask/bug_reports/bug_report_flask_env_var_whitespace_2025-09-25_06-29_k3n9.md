# Bug Report: Flask Environment Variable Parsers Don't Strip Whitespace

**Target**: `flask.helpers.get_debug_flag` and `flask.helpers.get_load_dotenv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The environment variable parsing functions `get_debug_flag()` and `get_load_dotenv()` fail to strip whitespace from environment variable values, causing unexpected behavior when users accidentally include leading or trailing spaces.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask.helpers import get_debug_flag, get_load_dotenv
import os


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  "])
)
def test_debug_flag_whitespace_stripping(value, whitespace):
    os.environ['FLASK_DEBUG'] = whitespace + value
    result_leading = get_debug_flag()

    os.environ['FLASK_DEBUG'] = value
    expected = get_debug_flag()

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )


@given(
    st.sampled_from(["false", "False", "no", "No", "0"]),
    st.sampled_from([" ", "\t", "  "])
)
def test_load_dotenv_whitespace_stripping(value, whitespace):
    os.environ['FLASK_SKIP_DOTENV'] = whitespace + value
    result_leading = get_load_dotenv(True)

    os.environ['FLASK_SKIP_DOTENV'] = value
    expected = get_load_dotenv(True)

    assert result_leading == expected, (
        f"Leading whitespace changed result: {whitespace + value!r} → {result_leading}, "
        f"expected {expected}"
    )
```

**Failing inputs**:
- `get_debug_flag()`: `FLASK_DEBUG=' false'` or `'false '`
- `get_load_dotenv()`: `FLASK_SKIP_DOTENV=' false'` or `'false '`

## Reproducing the Bug

```python
import os
from flask.helpers import get_debug_flag, get_load_dotenv

os.environ['FLASK_DEBUG'] = 'false'
print(get_debug_flag())

os.environ['FLASK_DEBUG'] = ' false'
print(get_debug_flag())

assert get_debug_flag() == True

os.environ['FLASK_SKIP_DOTENV'] = 'false'
print(get_load_dotenv(True))

os.environ['FLASK_SKIP_DOTENV'] = ' false'
print(get_load_dotenv(True))

assert get_load_dotenv(True) == False
```

Expected for `get_debug_flag()`: Both should return `False`
Actual: `'false'` → `False`, `' false'` → `True`

Expected for `get_load_dotenv()`: Both should return `True`
Actual: `'false'` → `True`, `' false'` → `False`

## Why This Is A Bug

Users commonly set environment variables with accidental whitespace, especially in shell scripts or `.env` files:

```bash
export FLASK_DEBUG="false "
export FLASK_SKIP_DOTENV=" no"
```

The current implementation does not strip this whitespace, causing:
- `FLASK_DEBUG=" false"` enables debug mode (expected: disabled)
- `FLASK_SKIP_DOTENV=" false"` skips dotenv loading (expected: load)

This violates the principle of least surprise and is inconsistent with how most configuration parsers handle environment variables.

## Fix

Strip whitespace from environment variable values before parsing:

```diff
 def get_debug_flag() -> bool:
     val = os.environ.get("FLASK_DEBUG")
+    if val:
+        val = val.strip()
     return bool(val and val.lower() not in {"0", "false", "no"})


 def get_load_dotenv(default: bool = True) -> bool:
     val = os.environ.get("FLASK_SKIP_DOTENV")

     if not val:
         return default
+
+    val = val.strip()
     return val.lower() in ("0", "false", "no")
```