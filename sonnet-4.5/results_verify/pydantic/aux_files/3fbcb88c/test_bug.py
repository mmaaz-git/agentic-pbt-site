#!/usr/bin/env python3
"""Test the reported bug in pydantic.deprecated.parse.load_str_bytes"""

from pydantic.deprecated.parse import load_str_bytes
import json

# Test 1: Basic reproduction test
print("=== Test 1: Basic Reproduction ===")
data = '{"key": "value"}'

try:
    result1 = load_str_bytes(data, content_type='application/json')
    print(f"Without charset: {result1}")
except Exception as e:
    print(f"Error without charset: {e}")

try:
    result2 = load_str_bytes(data, content_type='application/json; charset=utf-8')
    print(f"With charset: {result2}")
except TypeError as e:
    print(f"TypeError with charset: {e}")
except Exception as e:
    print(f"Other error with charset: {e}")

# Test 2: Other content types with parameters
print("\n=== Test 2: Other Content-Types with Parameters ===")

test_cases = [
    ('application/json', 'Basic JSON'),
    ('application/json; charset=utf-8', 'JSON with charset UTF-8'),
    ('application/json;charset=utf-8', 'JSON with charset no space'),
    ('application/json; charset=ISO-8859-1', 'JSON with different charset'),
    ('text/javascript', 'JavaScript type'),
    ('text/javascript; charset=utf-8', 'JavaScript with charset'),
    ('application/javascript', 'Application JavaScript'),
    ('application/javascript; charset=utf-8', 'Application JavaScript with charset'),
]

for content_type, description in test_cases:
    try:
        result = load_str_bytes(data, content_type=content_type)
        print(f"✓ {description}: Success - {result}")
    except TypeError as e:
        print(f"✗ {description}: TypeError - {e}")
    except Exception as e:
        print(f"✗ {description}: Other error - {e}")

# Test 3: Property-based test from the bug report
print("\n=== Test 3: Property-based test simulation ===")
test_dicts = [
    {},
    {"a": 1},
    {"foo": 42, "bar": 100},
    {"nested": {"key": "value"}},
]

for test_dict in test_dicts:
    json_str = json.dumps(test_dict)

    try:
        result1 = load_str_bytes(json_str, content_type='application/json')
        assert result1 == test_dict
        print(f"✓ Dict {test_dict}: Success without charset")
    except Exception as e:
        print(f"✗ Dict {test_dict}: Failed without charset - {e}")

    try:
        result2 = load_str_bytes(json_str, content_type='application/json; charset=utf-8')
        assert result2 == test_dict
        print(f"✓ Dict {test_dict}: Success with charset")
    except TypeError as e:
        print(f"✗ Dict {test_dict}: TypeError with charset - {e}")
    except Exception as e:
        print(f"✗ Dict {test_dict}: Other error with charset - {e}")