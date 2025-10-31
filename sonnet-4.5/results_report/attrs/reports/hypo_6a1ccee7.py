from hypothesis import given, strategies as st
from collections import namedtuple
import attr
from attr import asdict

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=5))
def test_asdict_with_list_of_namedtuples(coords):
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)
    result = asdict(c, retain_collection_types=True)

    assert isinstance(result['points'], list)
    assert len(result['points']) == len(points)

# Run the test
if __name__ == "__main__":
    test_asdict_with_list_of_namedtuples()