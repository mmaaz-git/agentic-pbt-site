from hypothesis import given, strategies as st, settings
from pandas.core.indexers.utils import length_of_indexer

@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.one_of(st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0), st.none())
)
@settings(max_examples=500)
def test_length_of_indexer_slice_property(start, stop, step):
    target = list(range(50))
    slc = slice(start, stop, step)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Failed for slice({start}, {stop}, {step}): computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    test_length_of_indexer_slice_property()