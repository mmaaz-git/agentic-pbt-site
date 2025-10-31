from hypothesis import given, strategies as st, assume
import pandas.io.common as pd_common

@given(
    names=st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20)
)
def test_dedup_names_multiindex_with_non_tuples_and_duplicates(names):
    assume(len(names) != len(set(names)))

    result = pd_common.dedup_names(names, is_potential_multiindex=True)
    result_list = list(result)

    assert len(result_list) == len(names)
    assert len(result_list) == len(set(result_list))

if __name__ == "__main__":
    test_dedup_names_multiindex_with_non_tuples_and_duplicates()