from hypothesis import given, strategies as st, settings
import pandas as pd
import pyarrow as pa


@settings(max_examples=500)
@given(
    st.lists(
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10),
        min_size=1,
        max_size=20
    ),
    st.integers(min_value=0, max_value=9)
)
def test_list_accessor_getitem_returns_correct_element(lists_of_ints, index):
    s = pd.Series(lists_of_ints, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    result = s.list[index]

    expected = [lst[index] if index < len(lst) else None for lst in lists_of_ints]

    for i, (res, exp) in enumerate(zip(result, expected)):
        if exp is None:
            assert pd.isna(res)
        else:
            assert res == exp


if __name__ == "__main__":
    test_list_accessor_getitem_returns_correct_element()