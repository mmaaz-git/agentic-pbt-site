import attr
from attr import asdict
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

# Create a container with a list containing a namedtuple
c = Container(points=[Point(1, 2)])

# This should crash with TypeError
result = asdict(c, retain_collection_types=True)
print(f"Result: {result}")