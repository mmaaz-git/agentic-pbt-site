from hypothesis import given, strategies as st
from pandas.core.indexers.utils import length_of_indexer

@given(
    indexer=st.slices(20),
    target_len=st.integers(min_value=0, max_value=20),
)
def test_length_of_indexer_slice_matches_actual(indexer, target_len):
    target = list(range(target_len))

    expected_len = length_of_indexer(indexer, target)
    actual_result = target[indexer]
    actual_len = len(actual_result)

    assert expected_len == actual_len, (
        f"length_of_indexer({indexer}, len={target_len}) = {expected_len}, "
        f"but len(target[indexer]) = {actual_len}"
    )

if __name__ == "__main__":
    test_length_of_indexer_slice_matches_actual()