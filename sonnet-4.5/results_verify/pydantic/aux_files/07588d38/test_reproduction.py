#!/usr/bin/env python3
"""Test to reproduce the whitespace bug in PYDANTIC_DISABLE_PLUGINS"""

# First, let's test the simple reproduction case
print("=" * 50)
print("TEST 1: Simple reproduction case")
print("=" * 50)

disabled_plugins = 'my-plugin, another-plugin'
split_result = disabled_plugins.split(',')

print(f"Input string: {disabled_plugins!r}")
print(f"Split result: {split_result}")

try:
    assert 'another-plugin' in split_result
    print("✓ 'another-plugin' found in split_result")
except AssertionError:
    print("✗ 'another-plugin' NOT found in split_result")
    print(f"  Looking for: 'another-plugin'")
    print(f"  Actual values in list: {split_result}")

# Test the hypothesis-based property test
print("\n" + "=" * 50)
print("TEST 2: Hypothesis property test")
print("=" * 50)

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
        assert plugin_name in split_result, f"Plugin '{plugin_name}' not found in {split_result}"

# Run with the specific failing example
print("\nTesting with the reported failing input:")
plugin_names = ['0', '00']
add_spaces = True

if add_spaces:
    disable_string = ', '.join(plugin_names)
else:
    disable_string = ','.join(plugin_names)

split_result = disable_string.split(',')
print(f"Plugin names: {plugin_names}")
print(f"Add spaces: {add_spaces}")
print(f"Disable string: {disable_string!r}")
print(f"Split result: {split_result}")

for plugin_name in plugin_names:
    if plugin_name in split_result:
        print(f"✓ Plugin '{plugin_name}' found")
    else:
        print(f"✗ Plugin '{plugin_name}' NOT found")
        print(f"  Looking for: {plugin_name!r}")
        print(f"  List contains: {[item for item in split_result]}")

# Let's also test what happens with the actual code pattern from pydantic
print("\n" + "=" * 50)
print("TEST 3: Simulating pydantic's actual code behavior")
print("=" * 50)

# Simulate what pydantic does
def simulate_pydantic_check(disabled_plugins_env, plugin_name):
    """Simulates the check in pydantic's _loader.py line 45"""
    if disabled_plugins_env is not None and plugin_name in disabled_plugins_env.split(','):
        return True  # Plugin is disabled
    return False  # Plugin is NOT disabled

test_cases = [
    ('plugin1,plugin2', 'plugin1', True),
    ('plugin1,plugin2', 'plugin2', True),
    ('plugin1, plugin2', 'plugin1', True),
    ('plugin1, plugin2', 'plugin2', False),  # This should be True but returns False due to space
    ('plugin1 , plugin2', 'plugin1', False),  # This should be True but returns False due to space
    ('plugin1 , plugin2', 'plugin2', False),  # This should be True but returns False due to space
]

for env_value, plugin_to_check, expected in test_cases:
    result = simulate_pydantic_check(env_value, plugin_to_check)
    status = "✓" if result == expected else "✗"
    print(f"{status} PYDANTIC_DISABLE_PLUGINS='{env_value}' checking '{plugin_to_check}': ")
    print(f"   Expected disabled={expected}, Got disabled={result}")
    if result != expected:
        print(f"   Split result: {env_value.split(',')}")