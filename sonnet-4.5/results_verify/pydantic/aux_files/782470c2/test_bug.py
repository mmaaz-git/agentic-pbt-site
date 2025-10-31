#!/usr/bin/env python3
"""Test the whitespace handling bug in pydantic plugin loader"""

# First, let's reproduce the basic parsing issue
print("=== Testing basic comma-separated parsing with spaces ===")
disabled_str = 'myplugin, yourplugin, theirplugin'
parsed_names = disabled_str.split(',')

print(f"Original string: '{disabled_str}'")
print(f"Parsed names: {parsed_names}")

example_entry_point_names = ['myplugin', 'yourplugin', 'theirplugin']

for ep_name in example_entry_point_names:
    is_disabled = ep_name in parsed_names
    print(f"{ep_name}: {'DISABLED' if is_disabled else 'NOT DISABLED'}")

print("\n=== Verifying the issue ===")
print(f"'myplugin' in parsed_names: {('myplugin' in parsed_names)}")
print(f"'yourplugin' in parsed_names: {('yourplugin' in parsed_names)}")
print(f"' yourplugin' in parsed_names: {(' yourplugin' in parsed_names)}")

# Now test with the actual function
print("\n=== Testing with actual pydantic function ===")
import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

# Test the actual behavior
original_env = os.environ.get('PYDANTIC_DISABLE_PLUGINS')
try:
    # Test with spaces after commas
    test_value = 'plugin1, plugin2, plugin3'
    print(f"Test PYDANTIC_DISABLE_PLUGINS='{test_value}'")

    # Simulate what the code does
    parsed = test_value.split(',')
    print(f"Split result: {parsed}")

    # Check if plugin names would match
    plugin_names = ['plugin1', 'plugin2', 'plugin3']
    for name in plugin_names:
        would_be_disabled = name in parsed
        print(f"  {name}: {'would be disabled' if would_be_disabled else 'would NOT be disabled'}")

    print("\n=== With stripped values ===")
    parsed_stripped = [name.strip() for name in test_value.split(',')]
    print(f"Split and stripped result: {parsed_stripped}")
    for name in plugin_names:
        would_be_disabled = name in parsed_stripped
        print(f"  {name}: {'would be disabled' if would_be_disabled else 'would NOT be disabled'}")

finally:
    if original_env is None:
        os.environ.pop('PYDANTIC_DISABLE_PLUGINS', None)
    else:
        os.environ['PYDANTIC_DISABLE_PLUGINS'] = original_env