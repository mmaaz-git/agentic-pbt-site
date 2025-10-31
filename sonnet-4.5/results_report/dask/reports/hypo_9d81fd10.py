from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.core import apply_filters

@given(st.lists(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.floats(allow_nan=False), st.text())
)))
def test_apply_filters_empty_filters_identity(parts):
    statistics = [{"columns": []} for _ in parts]
    filters = []
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    assert len(result_parts) == len(parts)
    assert len(result_stats) == len(statistics)

if __name__ == "__main__":
    # Run the test
    test_apply_filters_empty_filters_identity()