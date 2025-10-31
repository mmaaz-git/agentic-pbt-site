from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import apply_filters

@st.composite
def filter_with_null_count_data(draw):
    num_parts = draw(st.integers(min_value=1, max_value=10))
    col_name = "x"

    parts = []
    statistics = []

    for i in range(num_parts):
        min_val = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100)))
        max_val = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100)))

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val

        null_count = draw(st.integers(min_value=0, max_value=100))

        parts.append({"id": i})
        statistics.append({
            "filter": False,
            "columns": [{
                "name": col_name,
                "min": min_val,
                "max": max_val,
                "null_count": null_count
            }]
        })

    return parts, statistics, col_name

@given(filter_with_null_count_data())
@settings(max_examples=300)
def test_apply_filters_with_nulls_no_crash(data):
    parts, statistics, col_name = data

    filtered_parts, filtered_stats = apply_filters(
        parts, statistics, [(col_name, "=", 50)]
    )
    assert len(filtered_parts) <= len(parts)

if __name__ == "__main__":
    test_apply_filters_with_nulls_no_crash()