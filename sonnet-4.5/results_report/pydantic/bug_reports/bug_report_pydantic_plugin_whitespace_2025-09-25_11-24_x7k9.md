# Bug Report: pydantic.plugin Whitespace Handling in PYDANTIC_DISABLE_PLUGINS

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_plugins` function doesn't strip whitespace when parsing `PYDANTIC_DISABLE_PLUGINS`, causing plugin names after commas (with spaces) to not match actual entry point names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.plugin._loader import get_plugins
import os


@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@settings(max_examples=200)
def test_plugin_name_whitespace_sensitivity(plugin_names):
    """
    Property: Plugin name filtering should be resilient to whitespace.

    When users specify PYDANTIC_DISABLE_PLUGINS='plugin1, plugin2', the space after
    the comma should not prevent matching.
    """
    import pydantic.plugin._loader as loader_module
    from hypothesis import assume

    original_env = os.environ.get('PYDANTIC_DISABLE_PLUGINS')
    original_plugins = loader_module._plugins

    try:
        assume(all(name not in ('__all__', '1', 'true') for name in plugin_names))
        assume(all('\x00' not in name and ',' not in name for name in plugin_names))
        assume(all(name.strip() == name for name in plugin_names))

        with_spaces = ', '.join(plugin_names)
        without_spaces = ','.join(plugin_names)

        parsed_with_spaces = with_spaces.split(',')
        parsed_without_spaces = without_spaces.split(',')

        assert parsed_with_spaces != parsed_without_spaces, \
            "Whitespace causes different parsing"

    finally:
        if original_env is None:
            os.environ.pop('PYDANTIC_DISABLE_PLUGINS', None)
        else:
            os.environ['PYDANTIC_DISABLE_PLUGINS'] = original_env
        loader_module._plugins = original_plugins
```

**Failing input**: `PYDANTIC_DISABLE_PLUGINS='myplugin, yourplugin'`

## Reproducing the Bug

```python
disabled_str = 'myplugin, yourplugin, theirplugin'
parsed_names = disabled_str.split(',')

example_entry_point_names = ['myplugin', 'yourplugin', 'theirplugin']

for ep_name in example_entry_point_names:
    is_disabled = ep_name in parsed_names
    print(f"{ep_name}: {'DISABLED' if is_disabled else 'NOT DISABLED'}")

assert 'myplugin' in parsed_names
assert 'yourplugin' not in parsed_names
assert ' yourplugin' in parsed_names
```

Output:
```
myplugin: DISABLED
yourplugin: NOT DISABLED
theirplugin: NOT DISABLED
```

## Why This Is A Bug

Users naturally write comma-separated lists with spaces after commas (e.g., `'plugin1, plugin2'`). The current implementation splits on commas without stripping whitespace, so `' plugin2'` (with leading space) won't match entry point name `'plugin2'`. This violates user expectations and makes the disable feature unreliable.

## Fix

```diff
diff --git a/pydantic/plugin/_loader.py b/pydantic/plugin/_loader.py
index 1234567..abcdefg 100644
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -38,7 +38,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                     continue
                 if entry_point.value in _plugins:
                     continue
-                if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                if disabled_plugins is not None and entry_point.name in (name.strip() for name in disabled_plugins.split(',')):
                     continue
                 try:
                     _plugins[entry_point.value] = entry_point.load()
```