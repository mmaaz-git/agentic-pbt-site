# Bug Report: pydantic.plugin __all__ Keyword Doesn't Work in Comma-Separated Lists

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `PYDANTIC_DISABLE_PLUGINS` contains `__all__` in a comma-separated list (e.g., `"__all__,plugin1"`), it fails to disable all plugins and instead treats `__all__` as a literal plugin name.

## Property-Based Test

```python
from unittest.mock import Mock, patch
import os

def test_all_keyword_should_work_in_list():
    mock_plugin1 = Mock()
    mock_plugin2 = Mock()

    mock_entry_point1 = Mock()
    mock_entry_point1.group = 'pydantic'
    mock_entry_point1.name = 'plugin1'
    mock_entry_point1.value = 'test.plugin1'
    mock_entry_point1.load.return_value = mock_plugin1

    mock_entry_point2 = Mock()
    mock_entry_point2.group = 'pydantic'
    mock_entry_point2.name = 'plugin2'
    mock_entry_point2.value = 'test.plugin2'
    mock_entry_point2.load.return_value = mock_plugin2

    mock_dist = Mock()
    mock_dist.entry_points = [mock_entry_point1, mock_entry_point2]

    with patch('importlib.metadata.distributions', return_value=[mock_dist]):
        from pydantic.plugin import _loader
        _loader._plugins = None

        with patch.dict(os.environ, {'PYDANTIC_DISABLE_PLUGINS': '__all__,plugin1'}, clear=False):
            from pydantic.plugin._loader import get_plugins
            plugins = list(get_plugins())

            assert len(plugins) == 0, f"Expected 0 plugins with '__all__,plugin1', but got {len(plugins)}"
```

**Failing input**: `'__all__,plugin1'`, `'plugin1,__all__'`

## Reproducing the Bug

```python
import os
from unittest.mock import Mock, patch

mock_plugin1 = Mock()
mock_plugin2 = Mock()

mock_entry_point1 = Mock()
mock_entry_point1.group = 'pydantic'
mock_entry_point1.name = 'plugin1'
mock_entry_point1.value = 'test.plugin1'
mock_entry_point1.load.return_value = mock_plugin1

mock_entry_point2 = Mock()
mock_entry_point2.group = 'pydantic'
mock_entry_point2.name = 'plugin2'
mock_entry_point2.value = 'test.plugin2'
mock_entry_point2.load.return_value = mock_plugin2

mock_dist = Mock()
mock_dist.entry_points = [mock_entry_point1, mock_entry_point2]

with patch('importlib.metadata.distributions', return_value=[mock_dist]):
    from pydantic.plugin._loader import get_plugins
    from pydantic.plugin import _loader

    _loader._plugins = None
    with patch.dict(os.environ, {'PYDANTIC_DISABLE_PLUGINS': '__all__'}, clear=False):
        plugins = list(get_plugins())
        print(f"'__all__' alone: {len(plugins)} plugins")

    _loader._plugins = None
    with patch.dict(os.environ, {'PYDANTIC_DISABLE_PLUGINS': '__all__,plugin1'}, clear=False):
        plugins = list(get_plugins())
        print(f"'__all__,plugin1': {len(plugins)} plugins")
```

Output:
```
'__all__' alone: 0 plugins
'__all__,plugin1': 1 plugins
```

## Why This Is A Bug

The `__all__` keyword is documented as a way to disable all plugins. When a user includes it in a comma-separated list, the reasonable expectation is that it still disables all plugins. The current implementation only recognizes `__all__` when it's the exact and only value, treating it as a literal plugin name otherwise.

This is counterintuitive and could lead to users believing they've disabled all plugins when they haven't.

## Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -26,12 +26,17 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
     """
     disabled_plugins = os.getenv('PYDANTIC_DISABLE_PLUGINS')
     global _plugins, _loading_plugins
     if _loading_plugins:
         # this happens when plugins themselves use pydantic, we return no plugins
         return ()
-    elif disabled_plugins in ('__all__', '1', 'true'):
+    elif disabled_plugins and (
+        disabled_plugins in ('__all__', '1', 'true')
+        or '__all__' in [p.strip() for p in disabled_plugins.split(',')]
+    ):
         return ()
     elif _plugins is None:
         _plugins = {}
```

This fix checks if `__all__` appears anywhere in the comma-separated list and treats it as a directive to disable all plugins.