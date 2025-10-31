from hypothesis import given, strategies as st, settings, example
from pandas.core.indexers.utils import length_of_indexer


@given(
    st.one_of(
        st.integers(min_value=0, max_value=100),
        st.builds(
            slice,
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0)),
        ),
        st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
        st.lists(st.booleans(), min_size=1, max_size=50),
    ),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
@example(indexer=slice(None, None, -1), target_len=1)  # Force testing the reported failing case
def test_length_of_indexer_matches_actual_length(indexer, target_len):
    target = list(range(target_len))

    if isinstance(indexer, list) and len(indexer) > 0 and isinstance(indexer[0], bool):
        if len(indexer) != target_len:
            return

    try:
        claimed_length = length_of_indexer(indexer, target)
        actual_result = target[indexer]
        actual_length = len(actual_result)
        assert claimed_length == actual_length, f"Claimed {claimed_length}, actual {actual_length} for indexer={indexer}, target_len={target_len}"
    except (IndexError, ValueError, TypeError, KeyError):
        pass


if __name__ == "__main__":
    test_length_of_indexer_matches_actual_length()