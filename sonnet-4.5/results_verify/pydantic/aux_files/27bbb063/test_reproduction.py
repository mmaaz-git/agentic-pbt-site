import os
from unittest.mock import Mock, patch

# Test the property-based test case
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

# Run the test
try:
    test_all_keyword_should_work_in_list()
    print("Test PASSED (unexpected - should have failed)")
except AssertionError as e:
    print(f"Test FAILED as expected: {e}")

# Now run the reproduction script
print("\n--- Reproduction Script Output ---")

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

    # Let's also test plugin1,__all__
    _loader._plugins = None
    with patch.dict(os.environ, {'PYDANTIC_DISABLE_PLUGINS': 'plugin1,__all__'}, clear=False):
        plugins = list(get_plugins())
        print(f"'plugin1,__all__': {len(plugins)} plugins")

    # Let's test what happens with regular plugin name filtering
    _loader._plugins = None
    with patch.dict(os.environ, {'PYDANTIC_DISABLE_PLUGINS': 'plugin1'}, clear=False):
        plugins = list(get_plugins())
        print(f"'plugin1' alone: {len(plugins)} plugins (should be 1 - only plugin2)")