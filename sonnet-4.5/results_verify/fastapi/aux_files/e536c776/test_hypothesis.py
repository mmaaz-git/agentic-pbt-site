"""Hypothesis property-based test for the bug"""
from hypothesis import given, strategies as st
from collections import namedtuple
import attr

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

@given(st.lists(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.tuples(st.integers(), st.integers()),
    min_size=1, max_size=5
)))
def test_namedtuple_in_nested_collections(dict_items):
    if not dict_items or not dict_items[0]:
        # Skip empty test cases
        return

    nested_dict = {k: Point(v[0], v[1]) for k, v in list(dict_items[0].items())[:1]}
    obj = Container(data=nested_dict)

    try:
        result = attr.asdict(obj, recurse=True, retain_collection_types=True)
        # If it works, verify the type is preserved
        for key, point in nested_dict.items():
            assert isinstance(result['data'][key], Point)
        print("Test passed with input:", nested_dict)
    except TypeError as e:
        print("Test failed with error:", str(e))
        print("Failed input:", nested_dict)
        return  # Don't assert failure to see all failing cases

# Run the test
print("Running hypothesis tests...")
test_namedtuple_in_nested_collections()
print("Done!")