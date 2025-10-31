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