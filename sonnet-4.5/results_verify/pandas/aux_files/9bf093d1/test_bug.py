import pandas.core.arrays as arr

# Test case from bug report
sparse = arr.SparseArray([], fill_value=0)
print("Created empty SparseArray")

try:
    density = sparse.density
    print(f"Density: {density}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred: {e}")
    import traceback
    traceback.print_exc()