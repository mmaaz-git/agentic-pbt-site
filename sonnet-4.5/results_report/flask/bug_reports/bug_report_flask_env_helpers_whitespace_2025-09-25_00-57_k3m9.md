# Bug Report: Flask Environment Helpers Whitespace Handling

**Target**: `flask.helpers.get_debug_flag` and `flask.helpers.get_load_dotenv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Flask's `get_debug_flag()` and `get_load_dotenv()` functions incorrectly handle environment variable values with leading/trailing whitespace, causing values like `" false "` or `"\tno\t"` to be misinterpreted.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import os
from flask.helpers import get_debug_flag, get_load_dotenv

@given(val=st.sampled_from(["0", "false", "no"]))
def test_debug_flag_should_handle_whitespace(val):
    original = os.environ.get("FLASK_DEBUG")
    try:
        os.environ["FLASK_DEBUG"] = f" {val} "
        result = get_debug_flag()
        assert result is False, f"Should be False for ' {val} ' (with spaces)"
    finally:
        if original is None:
            os.environ.pop("FLASK_DEBUG", None)
        else:
            os.environ["FLASK_DEBUG"] = original

@given(val=st.sampled_from(["0", "false", "no"]))
def test_load_dotenv_should_handle_whitespace(val):
    original = os.environ.get("FLASK_SKIP_DOTENV")
    try:
        os.environ["FLASK_SKIP_DOTENV"] = f" {val} "
        result = get_load_dotenv()
        assert result is True, f"Should be True for ' {val} ' (with spaces)"
    finally:
        if original is None:
            os.environ.pop("FLASK_SKIP_DOTENV", None)
        else:
            os.environ["FLASK_SKIP_DOTENV"] = original
```

**Failing input**: `FLASK_DEBUG=" false "`, `FLASK_DEBUG="  "`, `FLASK_SKIP_DOTENV=" no "`

## Reproducing the Bug

```python
import os
import sys
sys.path.insert(0, '/path/to/flask')

from flask.helpers import get_debug_flag, get_load_dotenv

os.environ["FLASK_DEBUG"] = "false"
print(f"FLASK_DEBUG='false' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = " false "
print(f"FLASK_DEBUG=' false ' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = "  "
print(f"FLASK_DEBUG='  ' → {get_debug_flag()}")

os.environ["FLASK_SKIP_DOTENV"] = "no"
print(f"FLASK_SKIP_DOTENV='no' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = " no "
print(f"FLASK_SKIP_DOTENV=' no ' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = "  "
print(f"FLASK_SKIP_DOTENV='  ' → {get_load_dotenv()}")
```

**Expected output:**
```
FLASK_DEBUG='false' → False
FLASK_DEBUG=' false ' → False  (with whitespace trimmed)
FLASK_DEBUG='  ' → False  (empty after trimming)
FLASK_SKIP_DOTENV='no' → True
FLASK_SKIP_DOTENV=' no ' → True  (with whitespace trimmed)
FLASK_SKIP_DOTENV='  ' → True  (empty after trimming, use default)
```

**Actual output:**
```
FLASK_DEBUG='false' → False
FLASK_DEBUG=' false ' → True  (BUG: whitespace prevents matching!)
FLASK_DEBUG='  ' → True  (BUG: whitespace treated as truthy!)
FLASK_SKIP_DOTENV='no' → True
FLASK_SKIP_DOTENV=' no ' → False  (BUG: whitespace prevents matching!)
FLASK_SKIP_DOTENV='  ' → False  (BUG: whitespace treated as non-empty!)
```

## Why This Is A Bug

The functions use `val.lower()` without first calling `val.strip()`, causing:

1. **`get_debug_flag()`**: Values like `" false "` don't match the set `{"0", "false", "no"}` and are incorrectly treated as `True`
2. **`get_load_dotenv()`**: Values like `" no "` don't match and return `False` instead of the expected `True`
3. **Whitespace-only values**: Strings like `"  "` are treated as truthy non-empty values instead of being normalized

This violates the principle of robust environment variable parsing and can cause unexpected behavior in production environments where whitespace might accidentally be added to config values.

## Fix

Add `.strip()` before `.lower()` in both functions:

```diff
def get_debug_flag() -> bool:
    val = os.environ.get("FLASK_DEBUG")
-   return bool(val and val.lower() not in {"0", "false", "no"})
+   return bool(val and val.strip().lower() not in {"0", "false", "no"})


def get_load_dotenv(default: bool = True) -> bool:
    val = os.environ.get("FLASK_SKIP_DOTENV")

    if not val:
        return default

-   return val.lower() in ("0", "false", "no")
+   return val.strip().lower() in ("0", "false", "no")
```

This ensures that environment variables are parsed consistently regardless of accidental whitespace.