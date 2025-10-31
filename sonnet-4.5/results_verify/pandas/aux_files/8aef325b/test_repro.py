import pandas as pd
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df_original = pd.DataFrame({'col': ['']})
    print(f"Original: {df_original.shape}")
    print(df_original)
    print(f"Original values: {df_original['col'].tolist()}")
    print(f"Original dtype: {df_original['col'].dtype}")

    df_original.to_excel(tmp_path, index=False)
    df_read = pd.read_excel(tmp_path)

    print(f"\nRead back: {df_read.shape}")
    print(df_read)
    if len(df_read) > 0:
        print(f"Read back values: {df_read['col'].tolist()}")
        print(f"Read back dtype: {df_read['col'].dtype}")
    print(f"\nData lost: {df_original.shape != df_read.shape}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)