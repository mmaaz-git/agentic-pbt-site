from hypothesis import given, strategies as st
from collections import namedtuple
import attr
from attr import asdict
import pytest

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

def test_specific_failing_input():
    """Test with the specific failing input from the bug report"""
    coords = [(0, 0)]
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)

    # This should fail with retain_collection_types=True
    try:
        result = asdict(c, retain_collection_types=True)
        print(f"Result: {result}")
        assert isinstance(result['points'], list)
        assert len(result['points']) == len(points)
        return True
    except TypeError as e:
        print(f"TypeError as expected: {e}")
        return False

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=5))
def test_asdict_with_list_of_namedtuples(coords):
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)

    # Test with retain_collection_types=True
    result = asdict(c, retain_collection_types=True)

    assert isinstance(result['points'], list)
    assert len(result['points']) == len(points)

if __name__ == "__main__":
    print("Testing with specific failing input [(0, 0)]...")
    passed = test_specific_failing_input()

    if not passed:
        print("\nBug confirmed - asdict() crashes with namedtuples when retain_collection_types=True")
    else:
        print("\nNo bug found - test passed")

    print("\nTesting without retain_collection_types (should work)...")
    coords = [(0, 0)]
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)
    result_without_retain = asdict(c)  # Without retain_collection_types
    print(f"Without retain_collection_types: {result_without_retain}")