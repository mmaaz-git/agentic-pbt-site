from hypothesis import given, assume, strategies as st
import dask.dataframe.io.parquet.core as core

@given(
    st.lists(st.text(), min_size=0, max_size=20),
    st.lists(st.dictionaries(st.text(), st.integers()), min_size=0, max_size=20)
)
def test_apply_filters_empty_filters_returns_all(parts, statistics):
    assume(len(parts) == len(statistics))

    filters = []

    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)

    assert filtered_parts == parts
    assert filtered_stats == statistics

if __name__ == "__main__":
    test_apply_filters_empty_filters_returns_all()