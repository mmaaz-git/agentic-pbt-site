#!/usr/bin/env python3
"""Debug how include() and exclude() work with generators."""

import attr
from attr.filters import include, exclude, _split_what

@attr.s
class MyClass:
    field1 = attr.ib()
    field2 = attr.ib()
    field3 = attr.ib()

# Test _split_what directly with generator
fields = ["field1", "field2"]
gen = (f for f in fields)

print("Direct _split_what test with generator:")
cls, names, attrs = _split_what(gen)
print(f"Classes: {cls}")
print(f"Names: {names}")
print(f"Attrs: {attrs}")

# Now let's trace through include() function
print("\nHow include() uses _split_what:")
fields = ["field1", "field2"]
filter_func = include(*fields)  # Note: * unpacks, so it becomes include("field1", "field2")

# Create an instance
obj = MyClass(field1="value1", field2="value2", field3="value3")
result = attr.asdict(obj, filter=filter_func)
print(f"Result: {result}")

# The bug occurs when we pass a generator to _split_what, but include() doesn't pass a generator
# include() passes `what` which is the tuple of arguments passed to include()
print("\nThe key difference:")
print("include(*fields) unpacks fields, so include receives ('field1', 'field2') as arguments")
print("Inside include(), _split_what(what) receives the tuple of arguments")
print("But if we pass a generator to _split_what directly, it fails")

# Test with mixed types
print("\nTesting with mixed types (types, strings, attrs):")
mixed = [int, str, "field1", "field2", float]
gen = (x for x in mixed)
cls, names, attrs = _split_what(gen)
print(f"Generator - Classes: {cls}, Names: {names}, Attrs: {attrs}")

cls2, names2, attrs2 = _split_what(mixed)
print(f"List - Classes: {cls2}, Names: {names2}, Attrs: {attrs2}")