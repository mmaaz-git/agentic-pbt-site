import os

test_values = ['true', 'True', 'TRUE', '1', '__all__']

for value in test_values:
    disabled_plugins = value

    if disabled_plugins in ('__all__', '1', 'true'):
        result = "Disables all plugins"
    else:
        result = "Does NOT disable all plugins"

    expected = "Disables" if value.lower() in ('true', '1', '__all__') else "Does NOT disable"

    print(f"PYDANTIC_DISABLE_PLUGINS='{value}': {result} (expected: {expected})")