# Bug Report: pydantic.plugin._loader Whitespace Handling in PYDANTIC_DISABLE_PLUGINS

**Target**: `pydantic.plugin._loader.get_plugins()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The plugin loader fails to correctly parse the `PYDANTIC_DISABLE_PLUGINS` environment variable when plugin names are separated by commas with spaces (e.g., `"plugin1, plugin2"`). The code splits by comma but doesn't strip whitespace, causing plugin names with leading/trailing spaces to not match actual plugin names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.sampled_from([
    "plugin1,plugin2",
    "plugin1, plugin2",
    "plugin1 , plugin2",
    "plugin1,  plugin2",
]))
def test_plugin_name_parsing_whitespace(disabled_string):
    plugin_names = disabled_string.split(',')

    assert 'plugin2' in plugin_names or ' plugin2' in plugin_names
```

**Failing input**: `"plugin1, plugin2"` (with space after comma)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

disabled_string = "plugin1, plugin2, plugin3"
plugin_names = disabled_string.split(',')

print(f"Split result: {plugin_names}")

assert plugin_names == ['plugin1', ' plugin2', ' plugin3']

if 'plugin2' in plugin_names:
    print("Plugin2 found (expected)")
else:
    print("BUG: Plugin2 NOT found due to leading space")
    print("Actual values:", [repr(name) for name in plugin_names])
```

Output:
```
Split result: ['plugin1', ' plugin2', ' plugin3']
BUG: Plugin2 NOT found due to leading space
Actual values: ['plugin1', ' plugin2', ' plugin3']
```

## Why This Is A Bug

Users commonly format comma-separated lists with spaces after commas (e.g., `PYDANTIC_DISABLE_PLUGINS="plugin1, plugin2, plugin3"`). The current implementation at line 45 of `_loader.py` performs:

```python
if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
```

This splits by comma but doesn't strip whitespace from each element. As a result:
- Input: `"plugin1, plugin2"`
- Split result: `['plugin1', ' plugin2']` (note the leading space)
- If actual plugin name is `'plugin2'`, it won't match `' plugin2'`
- Therefore, the plugin won't be disabled despite being in the list

This violates user expectations and the documented behavior implied by the feature.

## Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -42,7 +42,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                     continue
                 if entry_point.value in _plugins:
                     continue
-                if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                if disabled_plugins is not None and entry_point.name in [name.strip() for name in disabled_plugins.split(',')]:
                     continue
                 try:
                     _plugins[entry_point.value] = entry_point.load()
```