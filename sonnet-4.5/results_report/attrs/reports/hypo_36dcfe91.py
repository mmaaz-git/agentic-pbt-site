from attr.filters import _split_what
from hypothesis import given, strategies as st

@given(st.lists(st.one_of(
    st.sampled_from([int, str, float]),
    st.text(min_size=1, max_size=20)
), min_size=1, max_size=20))
def test_split_what_generator_vs_list(items):
    gen_classes, gen_names, gen_attrs = _split_what(x for x in items)
    list_classes, list_names, list_attrs = _split_what(items)

    assert gen_classes == list_classes
    assert gen_names == list_names
    assert gen_attrs == list_attrs

if __name__ == "__main__":
    test_split_what_generator_vs_list()