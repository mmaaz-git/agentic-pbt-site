from dask.dataframe.io.parquet.core import sorted_columns

stats = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 5, "max": 10}]},
]

result = sorted_columns(stats)
print("Result:", result)