from hypothesis import given, strategies as st
import dask.dataframe.io.parquet.core as parquet_core

@given(st.lists(
    st.fixed_dictionaries({
        'columns': st.lists(
            st.fixed_dictionaries({
                'name': st.text(min_size=1, max_size=20),
                'min': st.one_of(st.none(), st.integers(-1000, 1000)),
                'max': st.one_of(st.none(), st.integers(-1000, 1000))
            }),
            min_size=1,
            max_size=5
        )
    }),
    min_size=1,
    max_size=10
))
def test_sorted_columns_divisions_are_sorted(statistics):
    result = parquet_core.sorted_columns(statistics)
    for item in result:
        assert item['divisions'] == sorted(item['divisions'])

if __name__ == "__main__":
    test_sorted_columns_divisions_are_sorted()