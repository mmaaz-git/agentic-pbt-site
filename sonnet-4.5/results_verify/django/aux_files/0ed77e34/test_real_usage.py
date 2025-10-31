#!/usr/bin/env python3
"""Test if mixed-type dictionaries could realistically appear in Django migrations"""

from django.db.migrations.serializer import serializer_factory

# Realistic scenarios where mixed-type dictionaries might appear:

# 1. Field choices with mixed types (e.g., database IDs vs string codes)
MIXED_CHOICES = {
    1: 'Option 1',
    2: 'Option 2',
    'custom': 'Custom Option',
    'other': 'Other'
}

# 2. Field defaults or validators
MIXED_CONFIG = {
    'max_length': 100,
    'min_length': 10,
    0: 'zero_fallback',
    1: 'one_fallback'
}

# 3. Model Meta options or custom attributes
MIXED_META = {
    'verbose_name': 'Item',
    'ordering': ['name'],
    1: 'primary',
    2: 'secondary'
}

# Test if these would be serialized in migrations
test_values = [
    (MIXED_CHOICES, "Field choices with mixed int/str keys"),
    (MIXED_CONFIG, "Configuration dict with mixed keys"),
    (MIXED_META, "Meta options with mixed keys"),
    ({0: 'default', 'error': 'message'}, "Error codes mixed with strings"),
    ({True: 'enabled', 'false': 'disabled'}, "Boolean mixed with string keys")
]

print("Testing realistic Django use cases:\n")
for value, description in test_values:
    print(f"{description}:")
    print(f"  Value: {value}")
    try:
        serializer = serializer_factory(value)
        result, imports = serializer.serialize()
        print(f"  ✓ Serialized successfully: {result}")
    except TypeError as e:
        print(f"  ✗ Failed with TypeError: {e}")
    print()