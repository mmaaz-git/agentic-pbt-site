# Bug Report: pydantic.plugin get_plugins Case-Sensitive Environment Variable

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `PYDANTIC_DISABLE_PLUGINS` environment variable check is case-sensitive, only recognizing lowercase `'true'` but not common variations like `'True'` or `'TRUE'`. This is inconsistent with typical environment variable conventions where boolean values are case-insensitive.

## Property-Based Test

```python
from hypothesis import given, strategies as st


@given(st.sampled_from(['true', 'True', 'TRUE', '1', '__all__']))
def test_disable_all_plugins_case_insensitive(value):
    disabled_plugins = value

    is_truthy = value.lower() in ('true', '1', '__all__')

    actual_disabled = disabled_plugins in ('__all__', '1', 'true')

    assert actual_disabled == is_truthy, (
        f"PYDANTIC_DISABLE_PLUGINS='{value}' should disable all plugins "
        f"regardless of case, but case-sensitive check fails"
    )
```

**Failing input**: `'True'`, `'TRUE'`, or any capitalized variation

## Reproducing the Bug

```python
import os

test_values = ['true', 'True', 'TRUE', '1', '__all__']

for value in test_values:
    disabled_plugins = value

    if disabled_plugins in ('__all__', '1', 'true'):
        result = "Disables all plugins"
    else:
        result = "Does NOT disable all plugins"

    expected = "Disables" if value.lower() in ('true', '1', '__all__') else "Does NOT disable"

    print(f"PYDANTIC_DISABLE_PLUGINS='{value}': {result} (expected: {expected})")
```

**Output:**
```
PYDANTIC_DISABLE_PLUGINS='true': Disables all plugins (expected: Disables)
PYDANTIC_DISABLE_PLUGINS='True': Does NOT disable all plugins (expected: Disables)
PYDANTIC_DISABLE_PLUGINS='TRUE': Does NOT disable all plugins (expected: Disables)
PYDANTIC_DISABLE_PLUGINS='1': Disables all plugins (expected: Disables)
PYDANTIC_DISABLE_PLUGINS='__all__': Disables all plugins (expected: Disables)
```

## Why This Is A Bug

Line 32 in `_loader.py` performs a case-sensitive check:

```python
elif disabled_plugins in ('__all__', '1', 'true'):
    return ()
```

This violates common expectations for environment variables:
- Users typically set boolean env vars as `'True'`, `'TRUE'`, `'1'`, `'yes'`, etc.
- Most environment variable parsers treat these case-insensitively
- Shell scripts and configuration tools often use `True` (capital T) by default

**User impact**: A user who sets `export PYDANTIC_DISABLE_PLUGINS=True` (with capital T, which is common) will find that plugins are NOT disabled. This creates a confusing user experience where the environment variable appears to be ignored.

## Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -29,7 +29,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
     if _loading_plugins:
         # this happens when plugins themselves use pydantic, we return no plugins
         return ()
-    elif disabled_plugins in ('__all__', '1', 'true'):
+    elif disabled_plugins and disabled_plugins.lower() in ('__all__', '1', 'true', 'yes'):
         return ()
     elif _plugins is None:
         _plugins = {}
```

Alternatively, for more robust handling:
```python
elif disabled_plugins and disabled_plugins.strip().lower() in ('__all__', '1', 'true', 'yes', 'on'):
    return ()
```