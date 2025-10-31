#!/usr/bin/env python3
"""Test the _nicetitle function bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.plot.facetgrid import _nicetitle

# Test the specific failing case
print("Testing the specific failing case from bug report:")
result = _nicetitle('', 0, maxchar=1, template="{coord}={value}")
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"Assertion would fail: len(result) <= 1 is {len(result) <= 1}")
print()

# Test various edge cases
print("Testing various maxchar values:")
test_cases = [
    ('a', 'b', 1),
    ('a', 'b', 2),
    ('a', 'b', 3),
    ('a', 'b', 4),
    ('coord', 'value', 1),
    ('coord', 'value', 5),
    ('coord', 'value', 10),
    ('longcoordname', 'longvalue', 3),
]

for coord, value, maxchar in test_cases:
    result = _nicetitle(coord, value, maxchar, "{coord}={value}")
    print(f"coord={repr(coord)}, value={repr(value)}, maxchar={maxchar}")
    print(f"  Result: {repr(result)}, Length: {len(result)}, Within limit: {len(result) <= maxchar}")