import pandas as pd
import tempfile
import os

df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])
print(f"Original columns: {df.columns.tolist()}")
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    temp_path = f.name

df.to_json(temp_path, orient="records")

with open(temp_path) as f:
    json_content = f.read()
print(f"JSON content: {json_content}")

result = pd.read_json(temp_path, orient="records")
print(f"Result columns: {result.columns.tolist()}")
print(f"Result shape: {result.shape}")

# Cleanup
os.unlink(temp_path)