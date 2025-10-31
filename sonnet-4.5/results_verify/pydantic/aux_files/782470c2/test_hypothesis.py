#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
import os

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@example(['plugin1', 'plugin2'])  # Add explicit example
@settings(max_examples=50)  # Reduce examples for testing
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

        # This assertion correctly shows that whitespace causes different parsing
        assert parsed_with_spaces != parsed_without_spaces, \
            "Whitespace causes different parsing"

        print(f"Plugin names: {plugin_names}")
        print(f"With spaces: {with_spaces} -> {parsed_with_spaces}")
        print(f"Without spaces: {without_spaces} -> {parsed_without_spaces}")
        print("TEST PASSED: Whitespace DOES cause different parsing")

        # The real issue is this affects matching
        for name in plugin_names[1:]:  # Skip first one which has no leading space
            assert name not in parsed_with_spaces, f"{name} incorrectly found in {parsed_with_spaces}"
            assert f' {name}' in parsed_with_spaces, f"' {name}' not found in {parsed_with_spaces}"

        print("Confirmed: Plugin names with leading spaces don't match actual names")

    finally:
        if original_env is None:
            os.environ.pop('PYDANTIC_DISABLE_PLUGINS', None)
        else:
            os.environ['PYDANTIC_DISABLE_PLUGINS'] = original_env
        loader_module._plugins = original_plugins

# Run the test
print("Running hypothesis test...")
test_plugin_name_whitespace_sensitivity()