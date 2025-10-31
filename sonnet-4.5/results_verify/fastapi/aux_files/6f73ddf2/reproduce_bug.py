import attr
from attr import asdict
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

# Test the reported bug
c = Container(points=[Point(1, 2)])
print("Testing asdict with namedtuple in list and retain_collection_types=True...")
try:
    result = asdict(c, retain_collection_types=True)
    print(f"Success! Result: {result}")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    import traceback
    traceback.print_exc()