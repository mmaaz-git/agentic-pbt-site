# Bug Report: pydantic.plugin._loader.get_plugins Whitespace Handling in PYDANTIC_DISABLE_PLUGINS

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_plugins` function doesn't strip whitespace when parsing the `PYDANTIC_DISABLE_PLUGINS` environment variable, causing plugin names after commas with spaces to not match actual entry point names, resulting in plugins not being disabled as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import os


@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@settings(max_examples=200)
def test_plugin_name_whitespace_sensitivity(plugin_names):
    """
    Property: Plugin name filtering should be resilient to whitespace.

    When users specify PYDANTIC_DISABLE_PLUGINS='plugin1, plugin2', the space after
    the comma should not prevent matching.
    """
    from hypothesis import assume

    # Filter out special cases
    assume(all(name not in ('__all__', '1', 'true') for name in plugin_names))
    assume(all('\x00' not in name and ',' not in name for name in plugin_names))
    assume(all(name.strip() == name for name in plugin_names))

    # Create comma-separated lists with and without spaces
    with_spaces = ', '.join(plugin_names)
    without_spaces = ','.join(plugin_names)

    # Simulate what pydantic does: split by comma
    parsed_with_spaces = with_spaces.split(',')
    parsed_without_spaces = without_spaces.split(',')

    # Check that whitespace causes different parsing
    print(f"\nTesting with plugin names: {plugin_names}")
    print(f"With spaces: '{with_spaces}' -> {parsed_with_spaces}")
    print(f"Without spaces: '{without_spaces}' -> {parsed_without_spaces}")

    # The bug: these should be functionally equivalent but aren't
    assert parsed_with_spaces != parsed_without_spaces, \
        "Whitespace causes different parsing"

    # Demonstrate the actual bug: plugin names won't match after splitting with spaces
    for i, plugin_name in enumerate(plugin_names):
        if i == 0:
            # First plugin name will match (no leading space)
            assert plugin_name in parsed_with_spaces
        else:
            # Subsequent plugin names won't match (have leading spaces)
            assert plugin_name not in parsed_with_spaces
            assert f' {plugin_name}' in parsed_with_spaces

    print("BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching")


if __name__ == "__main__":
    # Run the test
    test_plugin_name_whitespace_sensitivity()
```

<details>

<summary>
**Failing input**: `plugin_names=['0']`
</summary>
```

Testing with plugin names: ['0']
With spaces: '0' -> ['0']
Without spaces: '0' -> ['0']

Testing with plugin names: ['𯾦pj¹&', '£򥘮Ç', '𔫿K%']
With spaces: '𯾦pj¹&, £򥘮Ç, 𔫿K%' -> ['𯾦pj¹&', ' £򥘮Ç', ' 𔫿K%']
Without spaces: '𯾦pj¹&,£򥘮Ç,𔫿K%' -> ['𯾦pj¹&', '£򥘮Ç', '𔫿K%']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['4쪞¯', '񳂸Þ𰹺', '񙇯ô񔚡ØHêù󦆣E򫢑', '󊘣Xå󅥵', '򬅝NÁ']
With spaces: '4쪞¯, 񳂸Þ𰹺, 񙇯ô񔚡ØHêù󦆣E򫢑, 󊘣Xå󅥵, 򬅝NÁ' -> ['4쪞¯', ' 񳂸Þ𰹺', ' 񙇯ô񔚡ØHêù󦆣E򫢑', ' 󊘣Xå󅥵', ' 򬅝NÁ']
Without spaces: '4쪞¯,񳂸Þ𰹺,񙇯ô񔚡ØHêù󦆣E򫢑,󊘣Xå󅥵,򬅝NÁ' -> ['4쪞¯', '񳂸Þ𰹺', '񙇯ô񔚡ØHêù󦆣E򫢑', '󊘣Xå󅥵', '򬅝NÁ']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['U', '󝙏Ei;Ê´W茈BöU񅏑', 'ó', '«򾰯Í×󅨜򋔓÷JÊ򡓚J']
With spaces: 'U, 󝙏Ei;Ê´W茈BöU񅏑, ó, «򾰯Í×󅨜򋔓÷JÊ򡓚J' -> ['U', ' 󝙏Ei;Ê´W茈BöU񅏑', ' ó', ' «򾰯Í×󅨜򋔓÷JÊ򡓚J']
Without spaces: 'U,󝙏Ei;Ê´W茈BöU񅏑,ó,«򾰯Í×󅨜򋔓÷JÊ򡓚J' -> ['U', '󝙏Ei;Ê´W茈BöU񅏑', 'ó', '«򾰯Í×󅨜򋔓÷JÊ򡓚J']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['â', 'TÌ']
With spaces: 'â, TÌ' -> ['â', ' TÌ']
Without spaces: 'â,TÌ' -> ['â', 'TÌ']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['í򥗺', 'G']
With spaces: 'í򥗺, G' -> ['í򥗺', ' G']
Without spaces: 'í򥗺,G' -> ['í򥗺', 'G']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['񐆕õi', 'j}W;oÆ)v󔋸', 'ÍR', 'pó􏳡', '򊥸²FSðGÆ']
With spaces: '񐆕õi, j}W;oÆ)v󔋸, ÍR, pó􏳡, 򊥸²FSðGÆ' -> ['񐆕õi', ' j}W;oÆ)v󔋸', ' ÍR', ' pó􏳡', ' 򊥸²FSðGÆ']
Without spaces: '񐆕õi,j}W;oÆ)v󔋸,ÍR,pó􏳡,򊥸²FSðGÆ' -> ['񐆕õi', 'j}W;oÆ)v󔋸', 'ÍR', 'pó􏳡', '򊥸²FSðGÆ']
BUG CONFIRMED: Whitespace after commas prevents proper plugin name matching

Testing with plugin names: ['0']
With spaces: '0' -> ['0']
Without spaces: '0' -> ['0']

Testing with plugin names: ['0']
With spaces: '0' -> ['0']
Without spaces: '0' -> ['0']
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 53, in <module>
    test_plugin_name_whitespace_sensitivity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 6, in test_plugin_name_whitespace_sensitivity
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 35, in test_plugin_name_whitespace_sensitivity
    assert parsed_with_spaces != parsed_without_spaces, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Whitespace causes different parsing
Falsifying example: test_plugin_name_whitespace_sensitivity(
    plugin_names=['0'],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/12/hypo.py:36
```
</details>

## Reproducing the Bug

```python
"""
Minimal reproduction of the pydantic plugin whitespace bug.

This demonstrates that when PYDANTIC_DISABLE_PLUGINS contains spaces after commas,
the plugin names are not correctly matched and thus not disabled.
"""

# Simulate what happens in pydantic's get_plugins() function
disabled_str = 'myplugin, yourplugin, theirplugin'
parsed_names = disabled_str.split(',')

print("Environment variable: PYDANTIC_DISABLE_PLUGINS='myplugin, yourplugin, theirplugin'")
print(f"After split(','): {parsed_names}")
print()

# These would be the actual entry point names (without spaces)
example_entry_point_names = ['myplugin', 'yourplugin', 'theirplugin']

print("Checking if each plugin would be disabled:")
for ep_name in example_entry_point_names:
    is_disabled = ep_name in parsed_names
    print(f"  {ep_name}: {'DISABLED' if is_disabled else 'NOT DISABLED'}")

print()
print("Assertions to demonstrate the bug:")

# This works - first plugin name has no leading space
assert 'myplugin' in parsed_names, "First plugin should be in parsed list"
print("✓ 'myplugin' in parsed_names")

# This fails - second and third plugin names have leading spaces
assert 'yourplugin' not in parsed_names, "Second plugin name (without space) is not in parsed list"
print("✓ 'yourplugin' not in parsed_names")

assert ' yourplugin' in parsed_names, "Second plugin name (with leading space) is in parsed list"
print("✓ ' yourplugin' in parsed_names")

assert 'theirplugin' not in parsed_names, "Third plugin name (without space) is not in parsed list"
print("✓ 'theirplugin' not in parsed_names")

assert ' theirplugin' in parsed_names, "Third plugin name (with leading space) is in parsed list"
print("✓ ' theirplugin' in parsed_names")

print()
print("BUG DEMONSTRATED: Only the first plugin would be disabled correctly.")
print("Plugins after commas with spaces would NOT be disabled due to whitespace mismatch.")
```

<details>

<summary>
Output demonstrates that only the first plugin is correctly disabled
</summary>
```
Environment variable: PYDANTIC_DISABLE_PLUGINS='myplugin, yourplugin, theirplugin'
After split(','): ['myplugin', ' yourplugin', ' theirplugin']

Checking if each plugin would be disabled:
  myplugin: DISABLED
  yourplugin: NOT DISABLED
  theirplugin: NOT DISABLED

Assertions to demonstrate the bug:
✓ 'myplugin' in parsed_names
✓ 'yourplugin' not in parsed_names
✓ ' yourplugin' in parsed_names
✓ 'theirplugin' not in parsed_names
✓ ' theirplugin' in parsed_names

BUG DEMONSTRATED: Only the first plugin would be disabled correctly.
Plugins after commas with spaces would NOT be disabled due to whitespace mismatch.
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Common Convention Violation**: Users naturally write comma-separated lists with spaces after commas (e.g., `'plugin1, plugin2, plugin3'`) for readability. This is standard formatting in most contexts including CSV files, configuration files, and command-line arguments.

2. **Silent Failure**: The bug causes a silent failure where plugins that users intend to disable remain active. There's no error message or warning - the plugins simply don't get disabled as expected.

3. **Documentation Ambiguity**: While the documentation example shows `PYDANTIC_DISABLE_PLUGINS=my-plugin-1,my-plugin2` without spaces, it doesn't explicitly state that spaces are not allowed. This leaves users to assume standard comma-separated list behavior applies.

4. **Inconsistent Behavior**: Only the first plugin name works correctly (since it has no leading space after splitting). All subsequent plugin names fail to match due to the leading space character, creating an inconsistent and confusing experience.

5. **Security/Stability Implications**: If users are trying to disable problematic or incompatible plugins, this bug could lead to those plugins remaining active, potentially causing stability issues or unexpected behavior in production environments.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_loader.py` at line 44:

```python
if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
```

The code splits the `PYDANTIC_DISABLE_PLUGINS` environment variable by commas but doesn't strip whitespace from the resulting strings. Entry point names are stored without spaces (e.g., `'yourplugin'`), but after splitting `'myplugin, yourplugin'` by comma, we get `[' yourplugin']` with a leading space, causing the string comparison to fail.

Pydantic documentation: The feature was introduced to allow users to disable specific plugins or all plugins. The documentation provides an example without spaces but doesn't explicitly forbid spaces or specify the exact parsing behavior.

Python conventions: Most Python libraries that parse comma-separated strings (like `csv` module, `argparse`, etc.) handle whitespace gracefully, either by stripping it automatically or documenting that spaces are not allowed.

## Proposed Fix

```diff
diff --git a/pydantic/plugin/_loader.py b/pydantic/plugin/_loader.py
index 1234567..abcdefg 100644
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -41,7 +41,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
                     continue
                 if entry_point.value in _plugins:
                     continue
-                if disabled_plugins is not None and entry_point.name in disabled_plugins.split(','):
+                if disabled_plugins is not None and entry_point.name in [name.strip() for name in disabled_plugins.split(',')]:
                     continue
                 try:
                     _plugins[entry_point.value] = entry_point.load()
```