#!/usr/bin/env python3
"""Simple demonstration of the bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

print("=== Direct Test of Bug Report Example ===")

# Example from bug report
disabled_str = 'myplugin, yourplugin, theirplugin'
parsed_names = disabled_str.split(',')

print(f"PYDANTIC_DISABLE_PLUGINS would be: '{disabled_str}'")
print(f"After split(','), we get: {parsed_names}")
print()

# These are the actual plugin names from entry points
example_entry_point_names = ['myplugin', 'yourplugin', 'theirplugin']

print("Checking if each plugin would be disabled:")
for ep_name in example_entry_point_names:
    is_disabled = ep_name in parsed_names
    print(f"  {ep_name}: {'DISABLED' if is_disabled else 'NOT DISABLED'}")

print("\n=== Bug Analysis ===")
print(f"The issue: 'yourplugin' not in {parsed_names}")
print(f"Because: 'yourplugin' != ' yourplugin' (note the leading space)")

# Show the fix
print("\n=== With the proposed fix (strip whitespace) ===")
parsed_names_fixed = [name.strip() for name in disabled_str.split(',')]
print(f"After split(',') and strip(): {parsed_names_fixed}")

print("\nChecking if each plugin would be disabled with fix:")
for ep_name in example_entry_point_names:
    is_disabled = ep_name in parsed_names_fixed
    print(f"  {ep_name}: {'DISABLED' if is_disabled else 'NOT DISABLED'}")

print("\n=== Conclusion ===")
print("The bug is real: spaces after commas prevent plugin name matching")