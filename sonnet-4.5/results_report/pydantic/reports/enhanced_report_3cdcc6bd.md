# Bug Report: pydantic.plugin._loader Whitespace Mishandling in Plugin Disabling

**Target**: `pydantic.plugin._loader.get_plugins()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pydantic plugin loader fails to strip whitespace from plugin names when parsing the `PYDANTIC_DISABLE_PLUGINS` environment variable, causing plugins to not be disabled when users naturally format their comma-separated list with spaces after commas.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to demonstrate the whitespace bug
in pydantic's plugin loader PYDANTIC_DISABLE_PLUGINS parsing.
"""

from hypothesis import given, strategies as st, settings, example


@settings(max_examples=100)
@given(st.sampled_from([
    "plugin1,plugin2",      # Works: no spaces
    "plugin1, plugin2",     # Fails: space after comma
    "plugin1 , plugin2",    # Fails: space before and after comma
    "plugin1,  plugin2",   # Fails: two spaces after comma
]))
def test_plugin_name_parsing_whitespace(disabled_string):
    """Test that 'plugin2' can be found regardless of whitespace formatting."""
    plugin_names = disabled_string.split(',')

    # The test checks if 'plugin2' (without spaces) can be found
    # OR if ' plugin2' (with leading space) is in the list
    # This assertion is designed to always pass, but reveals the bug
    assert 'plugin2' in plugin_names or ' plugin2' in plugin_names

    # The real issue: if we're looking for 'plugin2' specifically
    # (as the plugin loader would), it fails when there's whitespace
    if 'plugin2' not in plugin_names:
        print(f"\nBUG FOUND with input: {repr(disabled_string)}")
        print(f"Split result: {plugin_names}")
        print(f"'plugin2' not found, but {repr([n for n in plugin_names if 'plugin2' in n][0])} is present")
        # This demonstrates the bug - 'plugin2' != ' plugin2'
        raise AssertionError(f"Plugin 'plugin2' not found due to whitespace in: {repr(disabled_string)}")


if __name__ == "__main__":
    # Run the test
    test_plugin_name_parsing_whitespace()
```

<details>

<summary>
**Failing input**: `'plugin1, plugin2'` and `'plugin1,  plugin2'`
</summary>
```
BUG FOUND with input: 'plugin1, plugin2'
Split result: ['plugin1', ' plugin2']
'plugin2' not found, but ' plugin2' is present

BUG FOUND with input: 'plugin1, plugin2'
Split result: ['plugin1', ' plugin2']
'plugin2' not found, but ' plugin2' is present
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 38, in <module>
  |     test_plugin_name_parsing_whitespace()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 11, in test_plugin_name_parsing_whitespace
  |     @given(st.sampled_from([
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 24, in test_plugin_name_parsing_whitespace
    |     assert 'plugin2' in plugin_names or ' plugin2' in plugin_names
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_plugin_name_parsing_whitespace(
    |     disabled_string='plugin1,  plugin2',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 33, in test_plugin_name_parsing_whitespace
    |     raise AssertionError(f"Plugin 'plugin2' not found due to whitespace in: {repr(disabled_string)}")
    | AssertionError: Plugin 'plugin2' not found due to whitespace in: 'plugin1, plugin2'
    | Falsifying example: test_plugin_name_parsing_whitespace(
    |     disabled_string='plugin1, plugin2',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for pydantic plugin loader whitespace bug.
This demonstrates that the PYDANTIC_DISABLE_PLUGINS parser doesn't strip
whitespace from plugin names after splitting by comma.
"""

# Simulate what happens inside pydantic's _loader.py at line 45
disabled_string = "plugin1, plugin2, plugin3"
plugin_names = disabled_string.split(',')

print(f"Input string: {repr(disabled_string)}")
print(f"Split result: {plugin_names}")
print(f"Split result (with repr): {[repr(name) for name in plugin_names]}")
print()

# Check if 'plugin2' can be found (without leading space)
if 'plugin2' in plugin_names:
    print("✓ 'plugin2' found in the list (expected behavior)")
else:
    print("✗ BUG: 'plugin2' NOT found in the list")
    print("  This is because the actual value is ' plugin2' with a leading space")

print()

# Check what's actually in the list
if ' plugin2' in plugin_names:
    print("✓ ' plugin2' (with leading space) IS in the list")
    print("  This means a plugin named 'plugin2' would NOT be disabled")
    print("  because 'plugin2' != ' plugin2'")

print()
print("Impact: When users write PYDANTIC_DISABLE_PLUGINS=\"plugin1, plugin2, plugin3\"")
print("        (with spaces after commas, which is natural), the plugins won't")
print("        actually be disabled due to whitespace not being stripped.")
```

<details>

<summary>
Whitespace causes plugin names to mismatch, preventing disabling
</summary>
```
Input string: 'plugin1, plugin2, plugin3'
Split result: ['plugin1', ' plugin2', ' plugin3']
Split result (with repr): ["'plugin1'", "' plugin2'", "' plugin3'"]

✗ BUG: 'plugin2' NOT found in the list
  This is because the actual value is ' plugin2' with a leading space

✓ ' plugin2' (with leading space) IS in the list
  This means a plugin named 'plugin2' would NOT be disabled
  because 'plugin2' != ' plugin2'

Impact: When users write PYDANTIC_DISABLE_PLUGINS="plugin1, plugin2, plugin3"
        (with spaces after commas, which is natural), the plugins won't
        actually be disabled due to whitespace not being stripped.
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Common Convention**: Users naturally write comma-separated lists with spaces after commas (e.g., `"item1, item2, item3"`). This is standard formatting in virtually all contexts.

2. **Silent Failure**: When a user sets `PYDANTIC_DISABLE_PLUGINS="plugin1, plugin2"`, they expect both plugins to be disabled. Instead, only `plugin1` is disabled while `plugin2` remains active because the code looks for `'plugin2'` but finds `' plugin2'` (with a leading space).

3. **Documentation Gap**: While the pydantic documentation shows an example without spaces (`my-plugin-1,my-plugin2`), it doesn't explicitly warn that spaces will cause failures. Users reasonably expect whitespace to be handled gracefully.

4. **Security/Stability Risk**: Plugins that users explicitly tried to disable may still run, potentially causing unexpected behavior or security issues if a problematic plugin was meant to be disabled.

5. **Inconsistent with Best Practices**: Most environment variable parsers in other tools strip whitespace from comma-separated values. The current implementation at line 45 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_loader.py` simply splits without stripping.

## Relevant Context

- **Source Location**: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_loader.py:45`
- **Documentation**: [Pydantic Plugins Documentation](https://docs.pydantic.dev/latest/concepts/plugins/) shows the format as `my-plugin-1,my-plugin2` but doesn't explicitly forbid spaces
- **Feature Status**: The plugin system is marked as experimental and subject to change
- **Environment Variable**: `PYDANTIC_DISABLE_PLUGINS` accepts:
  - `__all__`, `1`, `true` - Disables all plugins
  - Comma-separated string - Disables specified plugin(s)

The bug occurs because the code performs a simple string split without trimming whitespace:
```python
if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
```

This means `"plugin1, plugin2"` becomes `['plugin1', ' plugin2']` where the second element has a leading space that prevents matching.

## Proposed Fix

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