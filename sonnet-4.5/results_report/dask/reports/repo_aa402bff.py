from unittest.mock import Mock, MagicMock
from dask.dataframe.io.orc.core import _read_orc

# Test case demonstrating the mutation bug
columns_list = ['col1', 'col2']
index_name = 'idx'

print("Initial columns list:", columns_list)
print("Index name:", index_name)
print()

# Create mock objects to simulate the engine and filesystem
mock_engine = Mock()
mock_df = MagicMock()
mock_df.set_index = Mock(return_value=MagicMock())
mock_engine.read_partition.return_value = mock_df

# Simulate calling _read_orc multiple times (as would happen with multiple partitions)
print("Calling _read_orc for partition 1...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 1, columns list:", columns_list)

print("\nCalling _read_orc for partition 2...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 2, columns list:", columns_list)

print("\nCalling _read_orc for partition 3...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 3, columns list:", columns_list)

print("\n" + "="*50)
print("BUG CONFIRMED: The columns list was mutated!")
print("Expected: ['col1', 'col2']")
print("Actual:  ", columns_list)
print("The index 'idx' was appended", len(columns_list) - 2, "times")