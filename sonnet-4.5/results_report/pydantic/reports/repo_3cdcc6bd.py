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