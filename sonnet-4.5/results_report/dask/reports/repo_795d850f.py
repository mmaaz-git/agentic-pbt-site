from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import pyarrow as pa

# Create a minimal test case showing the mutation bug
columns_list = ['col1', 'col2']
original = columns_list.copy()

print(f"Original columns list before call: {columns_list}")

# Set up dummy parameters for _read_orc
parts = [("dummy_path", [0])]
schema = {'col1': pa.int64(), 'col2': pa.int64()}

# Call _read_orc - this will fail because dummy_path doesn't exist,
# but the mutation happens before the file read
try:
    _read_orc(
        parts,
        engine=ArrowORCEngine,
        fs=None,
        schema=schema,
        index='col1',
        columns=columns_list
    )
except Exception as e:
    print(f"Expected error (file doesn't exist): {e}")

print(f"\nAfter calling _read_orc:")
print(f"Original copy:   {original}")
print(f"Modified list:   {columns_list}")
print(f"Lists are equal: {columns_list == original}")
print(f"Mutation occurred: {columns_list != original}")

# Demonstrate the specific mutation
if 'col1' in columns_list:
    count = columns_list.count('col1')
    print(f"\n'col1' appears {count} times in the mutated list")