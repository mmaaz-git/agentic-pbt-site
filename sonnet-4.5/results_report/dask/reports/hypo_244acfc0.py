from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns


@st.composite
def statistics_with_none(draw):
    """Generate statistics where some row groups may have None min/max"""
    num_row_groups = draw(st.integers(min_value=2, max_value=5))
    column_name = draw(st.text(alphabet='abc', min_size=1, max_size=3))

    stats = []
    for i in range(num_row_groups):
        has_stats = draw(st.booleans())
        if has_stats:
            min_val = draw(st.integers(min_value=0, max_value=100))
            max_val = draw(st.integers(min_value=min_val, max_value=min_val + 10))
            col_stats = {"name": column_name, "min": min_val, "max": max_val}
        else:
            col_stats = {"name": column_name, "min": None, "max": None}

        stats.append({"columns": [col_stats]})

    return stats


@given(stats=statistics_with_none())
@settings(max_examples=100)
def test_sorted_columns_handles_none_gracefully(stats):
    """
    Property: sorted_columns should handle None min/max values without crashing.
    """
    result = sorted_columns(stats)

    assert isinstance(result, list)

    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)


if __name__ == "__main__":
    # Run the test
    test_sorted_columns_handles_none_gracefully()