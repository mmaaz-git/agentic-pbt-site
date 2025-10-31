import attr
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

# Create an instance with a namedtuple nested in a dictionary
obj = Container(data={'key': Point(1, 2)})

# This should work but crashes
try:
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    print(f"Success: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()