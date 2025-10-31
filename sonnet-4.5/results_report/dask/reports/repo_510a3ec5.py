from dask.dataframe.io.parquet.utils import _normalize_index_columns

# Demonstrate the bug: when both user_columns and user_index are None,
# the function returns overlapping column and index names without validation
user_columns = None
data_columns = ['0']
user_index = None
data_index = ['0']

print("Input parameters:")
print(f"  user_columns={user_columns}")
print(f"  data_columns={data_columns}")
print(f"  user_index={user_index}")
print(f"  data_index={data_index}")
print()

column_names, index_names = _normalize_index_columns(
    user_columns, data_columns, user_index, data_index
)

print("Output:")
print(f"  column_names={column_names}")
print(f"  index_names={index_names}")
print()

intersection = set(column_names).intersection(set(index_names))
print(f"Intersection between columns and indices: {intersection}")
print()

if intersection:
    print("BUG CONFIRMED: Column and index names overlap!")
    print(f"The name(s) {intersection} appear in both columns and indices.")
else:
    print("No bug: Column and index names are disjoint.")

print("\n" + "="*60 + "\n")

# Contrast with user-specified overlap - this correctly raises an error
print("For comparison, when users explicitly specify overlapping names:")
print("  user_columns=['0']")
print("  data_columns=['0']")
print("  user_index=['0']")
print("  data_index=['0']")
print()

try:
    column_names, index_names = _normalize_index_columns(
        ['0'], ['0'], ['0'], ['0']
    )
    print(f"  Output: column_names={column_names}, index_names={index_names}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")