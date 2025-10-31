from hypothesis import given, strategies as st, example
from collections import namedtuple
import attr
from attr import asdict

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=5))
@example([(0, 0)])  # The reported failing input
def test_asdict_with_list_of_namedtuples(coords):
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)

    # Test with retain_collection_types=True
    result = asdict(c, retain_collection_types=True)

    assert isinstance(result['points'], list)
    assert len(result['points']) == len(points)

if __name__ == "__main__":
    # Run the test with the specific failing input
    print("Testing with coords=[(0, 0)]...")
    try:
        test_asdict_with_list_of_namedtuples([(0, 0)])
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nRunning full hypothesis test...")
    test_asdict_with_list_of_namedtuples()