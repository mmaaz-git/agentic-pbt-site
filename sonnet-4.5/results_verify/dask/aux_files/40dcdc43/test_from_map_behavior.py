#!/usr/bin/env python3
"""Test to see if dd.from_map would pass the same columns list to multiple calls"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from unittest.mock import Mock, MagicMock

# Test whether multiple partitions would see the same mutated list
def test_multiple_partitions():
    print("Testing what happens with multiple partitions:")
    print("="*60)

    columns_list = ['col1', 'col2']
    call_count = 0
    columns_history = []

    def mock_read_orc(parts, *, engine, fs, schema, index, columns=None):
        nonlocal call_count
        call_count += 1
        # Record the state of columns at entry
        if columns is not None:
            columns_history.append(f"Call {call_count} entry: {columns.copy()}")
        else:
            columns_history.append(f"Call {call_count} entry: None")

        # Simulate the actual _read_orc behavior
        if index is not None and columns is not None:
            columns.append(index)
            columns_history.append(f"Call {call_count} after append: {columns.copy()}")

        mock_df = MagicMock()
        if index:
            mock_df.set_index = Mock(return_value=MagicMock())
        return mock_df

    # Simulate dd.from_map calling _read_orc for multiple partitions
    parts_list = [{'part': 1}, {'part': 2}, {'part': 3}]

    print(f"Initial columns list: {columns_list}")
    print(f"Simulating {len(parts_list)} partition reads...")
    print()

    # This simulates how dd.from_map would call the function
    # In the real implementation, it passes the same columns object to each call
    for i, part in enumerate(parts_list):
        print(f"Partition {i+1}:")
        result = mock_read_orc(
            part,
            engine=Mock(),
            fs=Mock(),
            schema={},
            index='idx',
            columns=columns_list  # Same list object passed each time!
        )

    print("\nHistory of columns parameter:")
    for entry in columns_history:
        print(f"  {entry}")

    print(f"\nFinal columns list: {columns_list}")
    print(f"Expected: ['col1', 'col2']")
    print(f"List grew by {len(columns_list) - 2} elements!")

if __name__ == "__main__":
    test_multiple_partitions()