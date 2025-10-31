from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.utils import _normalize_index_columns

@given(
    st.one_of(st.none(), st.text(min_size=1, max_size=10), st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    st.one_of(
        st.none(),
        st.just(False),
        st.text(min_size=1, max_size=10),
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
    ),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
)
def test_normalize_index_columns_no_intersection(user_columns, data_columns, user_index, data_index):
    try:
        column_names, index_names = _normalize_index_columns(
            user_columns, data_columns, user_index, data_index
        )
        intersection = set(column_names).intersection(set(index_names))
        assert len(intersection) == 0
    except ValueError as e:
        if "must not intersect" in str(e):
            pass
        else:
            raise

# Run the test
if __name__ == "__main__":
    test_normalize_index_columns_no_intersection()