import pandas as pd
import tempfile
import json
import os

# Create empty DataFrame with columns
df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])
print(f"Original DataFrame:\n{df}")
print(f"Original columns: {df.columns.tolist()}")
print(f"Original shape: {df.shape}\n")

# Test different orient options
orient_options = ['records', 'split', 'table', 'columns', 'index', 'values']

for orient in orient_options:
    print(f"Testing orient='{orient}':")
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            temp_path = f.name

        df.to_json(temp_path, orient=orient)

        with open(temp_path) as f:
            json_content = f.read()
        print(f"  JSON content: {json_content[:200]}...")

        # Special handling for 'table' orient
        if orient == 'table':
            result = pd.read_json(temp_path, orient='table')
        else:
            result = pd.read_json(temp_path, orient=orient)

        print(f"  Result columns: {result.columns.tolist()}")
        print(f"  Result shape: {result.shape}")
        print(f"  Columns preserved: {df.columns.tolist() == result.columns.tolist()}")

        os.unlink(temp_path)
    except Exception as e:
        print(f"  Error: {e}")
    print()