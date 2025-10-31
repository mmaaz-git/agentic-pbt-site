from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns

@st.composite
def statistics_with_nones(draw):
    num_row_groups = draw(st.integers(min_value=1, max_value=10))
    col_name = "test_col"

    statistics = []
    for i in range(num_row_groups):
        has_min = draw(st.booleans())
        has_max = draw(st.booleans())

        min_val = draw(st.integers(min_value=-100, max_value=100)) if has_min else None
        max_val = draw(st.integers(min_value=-100, max_value=100)) if has_max else None

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val

        statistics.append({
            "columns": [{
                "name": col_name,
                "min": min_val,
                "max": max_val
            }]
        })

    return statistics, col_name

@given(statistics_with_nones())
@settings(max_examples=500)
def test_sorted_columns_none_handling(data):
    statistics, col_name = data
    result = sorted_columns(statistics, columns=[col_name])

    for item in result:
        divisions = item["divisions"]
        assert None not in divisions
        assert divisions == sorted(divisions)

if __name__ == "__main__":
    test_sorted_columns_none_handling()