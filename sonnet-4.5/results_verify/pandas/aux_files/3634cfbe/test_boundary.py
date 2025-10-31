#!/usr/bin/env python3

import pandas.io.json as json_module
import sys
import math

print(f"sys.float_info.max: {sys.float_info.max}")

# Test various values near the boundary
test_values = [
    1.7976931348623157e+308,  # sys.float_info.max
    1.7976931348e+308,
    1.7976931346e+308,
    1.7976931345e+308,  # The reported failing value
    1.7976931344e+308,
    1.7976931340e+308,
    1.797693134e+308,
    1.797693133e+308,
    1.797693132e+308,
    1.797693131e+308,
    1.797693130e+308,
    1.797693129e+308,
    1.79769312e+308,
    1.7976931e+308,
    1.797693e+308,
]

print("\nTesting ujson round-trip for values near sys.float_info.max:")
print("-" * 80)

for value in test_values:
    json_str = json_module.ujson_dumps(value)
    recovered = json_module.ujson_loads(json_str)
    is_preserved = (value == recovered) or math.isclose(value, recovered, rel_tol=1e-9)

    print(f"Value: {value:.15e}")
    print(f"  JSON: {json_str}")
    print(f"  Recovered: {recovered}")
    print(f"  Preserved: {is_preserved}")
    print(f"  Recovered is inf: {recovered == float('inf')}")
    print()