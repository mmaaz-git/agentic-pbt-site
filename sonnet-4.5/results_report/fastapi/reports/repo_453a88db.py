import dask.dataframe.io.parquet.core as parquet_core

statistics = [
    {'columns': [{'name': 'col1', 'min': None, 'max': None}]},
    {'columns': [{'name': 'col1', 'min': 0, 'max': None}]}
]

result = parquet_core.sorted_columns(statistics)
print("Result:", result)