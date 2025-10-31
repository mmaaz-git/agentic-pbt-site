from dask.dataframe.io.parquet.core import apply_filters

# Case 1: min is None, max has a value
parts = [{'id': 0}]
statistics = [{
    'filter': False,
    'columns': [{
        'name': 'x',
        'min': None,
        'max': 0,
        'null_count': 0
    }]
}]

print("Test case 1: min=None, max=0")
print("Input parts:", parts)
print("Input statistics:", statistics)
print("Filter: [('x', '=', 50)]")

try:
    result = apply_filters(parts, statistics, [('x', '=', 50)])
    print("Result:", result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Case 2: max is None, min has a value
parts2 = [{'id': 0}]
statistics2 = [{
    'filter': False,
    'columns': [{
        'name': 'x',
        'min': 0,
        'max': None,
        'null_count': 0
    }]
}]

print("Test case 2: min=0, max=None")
print("Input parts:", parts2)
print("Input statistics:", statistics2)
print("Filter: [('x', '=', 50)]")

try:
    result2 = apply_filters(parts2, statistics2, [('x', '=', 50)])
    print("Result:", result2)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")