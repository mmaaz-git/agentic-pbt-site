import os

# Test the case-sensitivity issue with PYDANTIC_DISABLE_PLUGINS
test_values = ['true', 'True', 'TRUE', '1', '__all__']

print("Testing PYDANTIC_DISABLE_PLUGINS case sensitivity:\n")

for value in test_values:
    # Simulate the exact check from pydantic/plugin/_loader.py line 32
    disabled_plugins = value

    if disabled_plugins in ('__all__', '1', 'true'):
        result = "DISABLES plugins"
    else:
        result = "DOES NOT disable plugins"

    # What users would reasonably expect
    expected = "DISABLES" if value.lower() in ('true', '1', '__all__') else "DOES NOT disable"

    match = "✓" if result.startswith(expected) else "✗"
    print(f"  {match} PYDANTIC_DISABLE_PLUGINS='{value}': {result}")

    if not result.startswith(expected):
        print(f"     Expected: {expected} plugins (case-insensitive check)")

print("\nBug: 'True' and 'TRUE' do not disable plugins despite being reasonable values.")