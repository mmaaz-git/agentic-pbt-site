# Bug Report: pydantic.plugin._loader Whitespace in PYDANTIC_DISABLE_PLUGINS

**Target**: `pydantic.plugin._loader.get_plugins()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `PYDANTIC_DISABLE_PLUGINS` environment variable parser does not strip whitespace when splitting the comma-separated list. This causes plugins to not be disabled when the list contains spaces after commas (e.g., `'plugin1, plugin2'`).

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    plugin_names=st.lists(
        st.text(
            alphabet=st.characters(blacklist_characters=[',', ' ', '\t', '\n']),
            min_size=1,
            max_size=20
        ),
        min_size=1,
        max_size=5,
        unique=True
    ),
    add_spaces=st.booleans()
)
def test_plugin_disable_list_parsing_property(plugin_names, add_spaces):
    if add_spaces:
        disable_string = ', '.join(plugin_names)
    else:
        disable_string = ','.join(plugin_names)

    split_result = disable_string.split(',')

    for plugin_name in plugin_names:
        assert plugin_name in split_result
```

**Failing input**: `plugin_names=['0', '00'], add_spaces=True`

This creates the disable string `'0, 00'` which splits to `['0', ' 00']`. The plugin name `'00'` is not in the list because it was stored with a leading space `' 00'`.

## Reproducing the Bug

```python
disabled_plugins = 'my-plugin, another-plugin'
split_result = disabled_plugins.split(',')

print(split_result)

assert 'another-plugin' in split_result
```

**Output:**
```
['my-plugin', ' another-plugin']
AssertionError
```

The string splits to `['my-plugin', ' another-plugin']`, and checking `'another-plugin' in split_result` returns `False` because the list contains `' another-plugin'` (with a leading space).

## Why This Is A Bug

In `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_loader.py` at line 45:

```python
if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
    continue
```

When users set `PYDANTIC_DISABLE_PLUGINS='plugin1, plugin2'` (a natural way to write comma-separated lists), the second plugin won't be disabled because `'plugin2'` is not in `['plugin1', ' plugin2']`.

This violates user expectations and the documented behavior of disabling specific plugins. Users would expect both whitespace-tolerant formats (`'plugin1,plugin2'` and `'plugin1, plugin2'`) to work equivalently.

## Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -42,7 +42,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                     continue
                 if entry_point.value in _plugins:
                     continue
-                if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                if disabled_plugins is not None and entry_point.name in [s.strip() for s in disabled_plugins.split(',')]:
                     continue
                 try:
                     _plugins[entry_point.value] = entry_point.load()
```

Alternatively, for better performance (avoiding list creation on each iteration):

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -33,6 +33,8 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
         return ()
     elif _plugins is None:
         _plugins = {}
+        disabled_set = set(s.strip() for s in disabled_plugins.split(',')) if disabled_plugins else set()
+
         # set _loading_plugins so any plugins that use pydantic don't themselves use plugins
         _loading_plugins = True
         try:
@@ -42,7 +44,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                     continue
                 if entry_point.value in _plugins:
                     continue
-                if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                if entry_point.name in disabled_set:
                     continue
                 try:
                     _plugins[entry_point.value] = entry_point.load()
```