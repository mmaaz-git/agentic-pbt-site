#!/usr/bin/env python3
"""Test how include() and exclude() work with generators."""

import attr
from attr.filters import include, exclude

@attr.s
class MyClass:
    field1 = attr.ib()
    field2 = attr.ib()
    field3 = attr.ib()

# Test with a generator expression
fields_to_include = ["field1", "field2"]
gen_filter = include(*(f for f in fields_to_include))

# Test with a list
list_filter = include(*fields_to_include)

# Create an instance
obj = MyClass(field1="value1", field2="value2", field3="value3")

print("Testing with generator-based filter:")
try:
    result_gen = attr.asdict(obj, filter=gen_filter)
    print(f"Result with generator filter: {result_gen}")
except Exception as e:
    print(f"Error with generator filter: {e}")

print("\nTesting with list-based filter:")
try:
    result_list = attr.asdict(obj, filter=list_filter)
    print(f"Result with list filter: {result_list}")
except Exception as e:
    print(f"Error with list filter: {e}")

# Also test exclude with generators
print("\nTesting exclude with generator:")
gen_exclude = exclude(*(f for f in ["field3"]))
try:
    result_gen_exclude = attr.asdict(obj, filter=gen_exclude)
    print(f"Result with generator exclude: {result_gen_exclude}")
except Exception as e:
    print(f"Error with generator exclude: {e}")