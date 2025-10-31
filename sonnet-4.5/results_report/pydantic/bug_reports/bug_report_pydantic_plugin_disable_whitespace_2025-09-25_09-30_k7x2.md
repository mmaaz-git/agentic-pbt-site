# Bug Report: pydantic.plugin Whitespace in PYDANTIC_DISABLE_PLUGINS

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_plugins` function fails to properly disable plugins when the `PYDANTIC_DISABLE_PLUGINS` environment variable contains spaces after commas, causing plugins to remain enabled when users expect them to be disabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5))
def test_plugin_name_matching_with_spaces(plugin_names):
    """Plugin names should match regardless of spaces after commas in the env var."""
    disabled_str_with_spaces = ', '.join(plugin_names)
    split_with_spaces = disabled_str_with_spaces.split(',')

    for i, name in enumerate(plugin_names):
        if i > 0:
            assert name not in split_with_spaces, f"Plugin '{name}' won't be matched due to leading space"
```

**Failing input**: `plugin_names = ['plugin1', 'plugin2']` where `disabled_str_with_spaces = 'plugin1, plugin2'`

## Reproducing the Bug

```python
disabled_plugins = 'plugin1, plugin2, plugin3'
plugins = ['plugin1', 'plugin2', 'plugin3']

for plugin_name in plugins:
    if plugin_name in disabled_plugins.split(','):
        print(f"Plugin '{plugin_name}' is disabled")
    else:
        print(f"Plugin '{plugin_name}' is NOT disabled")

split_result = disabled_plugins.split(',')
print('plugin2' in split_result)
print(' plugin2' in split_result)
```

Output:
```
Plugin 'plugin1' is disabled
Plugin 'plugin2' is NOT disabled
Plugin 'plugin3' is NOT disabled
False
True
```

## Why This Is A Bug

When users naturally write `PYDANTIC_DISABLE_PLUGINS=plugin1, plugin2, plugin3` (with spaces after commas for readability), the `split(',')` operation preserves those spaces, resulting in `['plugin1', ' plugin2', ' plugin3']`. Since plugin names don't have leading spaces, the string matching on line 45 of `_loader.py` fails for all plugins except the first one, leaving plugins enabled when they should be disabled.

## Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -42,7 +42,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                         continue
                     if entry_point.value in _plugins:
                         continue
-                    if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                    if disabled_plugins is not None and entry_point.name in [p.strip() for p in disabled_plugins.split(',')]:
                         continue
                     try:
                         _plugins[entry_point.value] = entry_point.load()
```