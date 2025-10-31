#!/usr/bin/env python3
"""Test to reproduce the reported bug about _read_orc mutating columns list"""

from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.io.orc.core import _read_orc

# First, let's run the exact reproduction case from the bug report
def test_exact_reproduction():
    columns_list = ['col1', 'col2']
    index_name = 'col1'

    print("Testing exact reproduction case from bug report:")
    print(f"Before: {columns_list}")

    mock_engine = Mock()
    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    _read_orc(
        parts=[],
        engine=mock_engine,
        fs=Mock(),
        schema={},
        index=index_name,
        columns=columns_list
    )

    print(f"After: {columns_list}")
    print(f"Was list mutated? {columns_list != ['col1', 'col2']}")
    return columns_list != ['col1', 'col2']

# Test the specific failing case mentioned
def test_failing_case():
    columns_list = ['0']
    index_name = '0'
    original = columns_list.copy()

    print("\nTesting specific failing case from bug report:")
    print(f"Before: {columns_list}")

    mock_engine = Mock()
    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    _read_orc(
        parts=[],
        engine=mock_engine,
        fs=Mock(),
        schema={},
        index=index_name,
        columns=columns_list
    )

    print(f"After: {columns_list}")
    print(f"Original: {original}")
    print(f"Was list mutated? {columns_list != original}")
    return columns_list != original

# Test when index is None
def test_index_none():
    columns_list = ['col1', 'col2']
    original = columns_list.copy()

    print("\nTesting when index is None:")
    print(f"Before: {columns_list}")

    mock_engine = Mock()
    mock_df = MagicMock()
    mock_engine.read_partition.return_value = mock_df

    _read_orc(
        parts=[],
        engine=mock_engine,
        fs=Mock(),
        schema={},
        index=None,
        columns=columns_list
    )

    print(f"After: {columns_list}")
    print(f"Was list mutated? {columns_list != original}")
    return columns_list != original

# Test when columns is None
def test_columns_none():
    print("\nTesting when columns is None:")

    mock_engine = Mock()
    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    try:
        _read_orc(
            parts=[],
            engine=mock_engine,
            fs=Mock(),
            schema={},
            index='some_index',
            columns=None
        )
        print("No error when columns is None")
    except Exception as e:
        print(f"Error occurred: {e}")

# Run the property-based test with hypothesis
@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1))
@settings(max_examples=50)  # Reduced for faster testing
def test_property_based(columns_list, index_name):
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

    assert columns_list == original_columns, f"columns list was mutated: {original_columns} -> {columns_list}"

if __name__ == "__main__":
    print("="*60)
    print("Testing _read_orc mutation bug")
    print("="*60)

    bug1 = test_exact_reproduction()
    bug2 = test_failing_case()
    bug3 = test_index_none()
    test_columns_none()

    print("\n" + "="*60)
    print("Summary:")
    print(f"Exact reproduction case shows mutation: {bug1}")
    print(f"Specific failing case shows mutation: {bug2}")
    print(f"Index=None case shows mutation: {bug3}")

    print("\n" + "="*60)
    print("Running property-based tests...")
    try:
        test_property_based()
        print("Property-based tests completed")
    except AssertionError as e:
        print(f"Property-based test failed: {e}")

    print("="*60)