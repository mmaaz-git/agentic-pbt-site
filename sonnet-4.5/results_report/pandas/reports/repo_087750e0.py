#!/usr/bin/env python3
"""Minimal reproduction of pandas ujson precision loss bug"""

from pandas.io.json import ujson_dumps, ujson_loads

# Test value with precision loss
test_value = 1.0000000000000002e+16

# Serialize and deserialize
json_str = ujson_dumps(test_value)
result = ujson_loads(json_str)

print(f"Original value:     {test_value}")
print(f"Original value (repr): {repr(test_value)}")
print(f"JSON string:        {json_str}")
print(f"Deserialized value: {result}")
print(f"Deserialized (repr):   {repr(result)}")
print(f"Values are equal:   {test_value == result}")
print(f"Difference:         {test_value - result}")