from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc

@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1))
@settings(max_examples=200)
def test_read_orc_does_not_mutate_columns(columns_list, index_name):
    original_columns = columns_list.copy()

    mock_engine = Mock()
    mock_engine.read_partition = Mock(return_value=Mock())

    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    try:
        _read_orc(
            parts=[],
            engine=mock_engine,
            fs=Mock(),
            schema={},
            index=index_name,
            columns=columns_list
        )
    except Exception:
        pass

    assert columns_list == original_columns, f"columns list should not be mutated. Original: {original_columns}, After: {columns_list}"

if __name__ == "__main__":
    test_read_orc_does_not_mutate_columns()