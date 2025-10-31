import pandas as pd
import tempfile
import os

df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])
print(f"Original index type: {type(df.index)}")
print(f"Original index: {df.index}")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

df.to_json(temp_path, orient="split")
result = pd.read_json(temp_path, orient="split")

print(f"Result index type: {type(result.index)}")
print(f"Result index: {result.index}")
print(f"Index types match: {type(df.index) == type(result.index)}")

# Clean up
os.unlink(temp_path)