from hypothesis import given, strategies as st, settings
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
@settings(max_examples=100)
def test_namedtuple_in_nested_collections(dict_items):
    if not dict_items or not dict_items[0]:
        return  # Skip empty cases

    nested_dict = {k: Point(v[0], v[1]) for k, v in list(dict_items[0].items())[:1]}
    obj = Container(data=nested_dict)
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    for key, point in nested_dict.items():
        assert isinstance(result['data'][key], Point)

if __name__ == "__main__":
    test_namedtuple_in_nested_collections()