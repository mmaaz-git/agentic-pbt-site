from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Create parts with None stripe values, as created by read_metadata
# when split_stripes=False (line 63 in arrow.py)
parts = [[("file1.orc", None)], [("file2.orc", None)]]

# This should work but will crash with TypeError
result = ArrowORCEngine._aggregate_files(
    aggregate_files=True,
    split_stripes=2,
    parts=parts
)

print("Result:", result)