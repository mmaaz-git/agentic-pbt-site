from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns
import string


@st.composite
def statistics_strategy(draw):
    num_row_groups = draw(st.integers(min_value=0, max_value=20))
    num_columns = draw(st.integers(min_value=1, max_value=5))

    column_names = [
        draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10))
        for _ in range(num_columns)
    ]

    stats = []
    for _ in range(num_row_groups):
        columns = []
        for col_name in column_names:
            has_stats = draw(st.booleans())
            if has_stats:
                min_val = draw(st.integers(min_value=-1000, max_value=1000) | st.none())
                if min_val is not None:
                    max_val = draw(
                        st.integers(min_value=min_val, max_value=1000) | st.none()
                    )
                else:
                    max_val = None
                columns.append({"name": col_name, "min": min_val, "max": max_val})
            else:
                columns.append({"name": col_name})

        stats.append({"columns": columns})

    return stats


@given(statistics_strategy())
@settings(max_examples=1000)
def test_sorted_columns_divisions_are_sorted(statistics):
    result = sorted_columns(statistics)
    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)

if __name__ == "__main__":
    test_sorted_columns_divisions_are_sorted()