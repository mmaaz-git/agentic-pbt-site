import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.sampled_from([
    "plugin1,plugin2",
    "plugin1, plugin2",
    "plugin1 , plugin2",
    "plugin1,  plugin2",
]))
def test_plugin_name_parsing_whitespace(disabled_string):
    plugin_names = disabled_string.split(',')

    assert 'plugin2' in plugin_names or ' plugin2' in plugin_names

    # Let's also print what we get for understanding
    print(f"Input: {repr(disabled_string)}")
    print(f"Split result: {plugin_names}")
    print(f"'plugin2' in result: {'plugin2' in plugin_names}")
    print(f"' plugin2' in result: {' plugin2' in plugin_names}")
    print("---")

if __name__ == "__main__":
    test_plugin_name_parsing_whitespace()