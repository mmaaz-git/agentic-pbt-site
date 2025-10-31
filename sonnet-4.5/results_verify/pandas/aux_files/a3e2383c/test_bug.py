import pandas as pd
import tempfile
import os

print("Testing double close() on ExcelWriter...")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    writer = pd.ExcelWriter(tmp_path, engine='openpyxl')
    df = pd.DataFrame({'A': [1, 2, 3]})
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    print("First close()...")
    writer.close()
    print("First close() succeeded")

    print("Second close()...")
    writer.close()
    print("Second close() succeeded")
except Exception as e:
    print(f"Error on second close(): {type(e).__name__}: {e}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)