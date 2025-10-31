"""Test script to reproduce the reported bug"""
import attr
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

# Test the bug
print("Testing namedtuple in nested dictionary with retain_collection_types=True")
obj = Container(data={'key': Point(1, 2)})
print(f"Object: {obj}")
print(f"Data type: {type(obj.data['key'])}")
print(f"Data value: {obj.data['key']}")

try:
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    print(f"Success! Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test without retain_collection_types
print("Testing same code without retain_collection_types=True")
try:
    result = attr.asdict(obj, recurse=True, retain_collection_types=False)
    print(f"Success! Result: {result}")
    print(f"Result data type: {type(result['data']['key'])}")
except TypeError as e:
    print(f"ERROR: {e}")