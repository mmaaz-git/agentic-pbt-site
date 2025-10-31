# Bug Report: Flask Environment Variable Whitespace Handling

**Target**: `flask.helpers.get_debug_flag` and `flask.helpers.get_load_dotenv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The environment variable parsing functions `get_debug_flag()` and `get_load_dotenv()` fail to strip whitespace from values before checking if they match the "falsy" values (`"0"`, `"false"`, `"no"`). This causes values with surrounding whitespace like `" false "` or `" 0 "` to be treated as truthy instead of falsy, leading to unexpected behavior.

## Property-Based Test

```python
import os
from hypothesis import given, strategies as st, example
from flask.helpers import get_debug_flag, get_load_dotenv

@given(
    st.sampled_from(["0", "false", "no"]),
    st.text(alphabet=" \t", min_size=1, max_size=5),
    st.text(alphabet=" \t", min_size=1, max_size=5)
)
@example("false", " ", " ")
def test_get_debug_flag_should_strip_whitespace(falsy_value, prefix, suffix):
    value_with_whitespace = prefix + falsy_value + suffix

    original_val = os.environ.get("FLASK_DEBUG")
    try:
        os.environ["FLASK_DEBUG"] = value_with_whitespace
        result = get_debug_flag()

        assert result is False, (
            f"FLASK_DEBUG={value_with_whitespace!r} should disable debug mode, "
            f"but got {result}"
        )
    finally:
        if original_val is None:
            os.environ.pop("FLASK_DEBUG", None)
        else:
            os.environ["FLASK_DEBUG"] = original_val
```

**Failing input**: `FLASK_DEBUG=" false "` (or any falsy value with whitespace)

## Reproducing the Bug

```python
import os
from flask.helpers import get_debug_flag, get_load_dotenv

os.environ["FLASK_DEBUG"] = " false "
result = get_debug_flag()
print(f"FLASK_DEBUG=' false ' returns: {result}")
assert result is True

os.environ["FLASK_DEBUG"] = " 0 "
result = get_debug_flag()
print(f"FLASK_DEBUG=' 0 ' returns: {result}")
assert result is True

os.environ["FLASK_SKIP_DOTENV"] = " false "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' false ' returns: {result}")
assert result is False

os.environ["FLASK_SKIP_DOTENV"] = " 0 "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' 0 ' returns: {result}")
assert result is False
```

## Why This Is A Bug

1. **Violates user expectations**: Users who set `FLASK_DEBUG=" false "` (with accidental whitespace from shell scripts, .env files, or configuration) would reasonably expect debug mode to be disabled, not enabled.

2. **Common in practice**: Whitespace is easily introduced in environment variables through shell quoting, configuration file formatting, or copy-paste errors. For example:
   - Shell: `export FLASK_DEBUG=" false "` (accidental space in quotes)
   - .env file: `FLASK_DEBUG = false` (space around equals sign)

3. **Security implications**: For `get_debug_flag()`, failing to disable debug mode when the user intended to can expose sensitive information in production.

4. **Inconsistent with principle of robustness**: Environment variable parsing should be forgiving of whitespace, which is a common source of human error.

5. **No documentation**: The docstrings don't mention that whitespace is significant, suggesting this behavior is unintentional.

## Fix

```diff
diff --git a/flask/helpers.py b/flask/helpers.py
index 1234567..abcdefg 100644
--- a/flask/helpers.py
+++ b/flask/helpers.py
@@ -30,7 +30,8 @@ def get_debug_flag() -> bool:
     :envvar:`FLASK_DEBUG` environment variable. The default is ``False``.
     """
     val = os.environ.get("FLASK_DEBUG")
-    return bool(val and val.lower() not in {"0", "false", "no"})
+    if val:
+        val = val.strip()
+    return bool(val and val.lower() not in {"0", "false", "no"})


 def get_load_dotenv(default: bool = True) -> bool:
@@ -43,7 +44,8 @@ def get_load_dotenv(default: bool = True) -> bool:
     :param default: What to return if the env var isn't set.
     """
     val = os.environ.get("FLASK_SKIP_DOTENV")
-
+    if val:
+        val = val.strip()
+
     if not val:
         return default

     return val.lower() in ("0", "false", "no")
```