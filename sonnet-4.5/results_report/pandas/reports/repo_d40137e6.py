import pandas as pd
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    writer = pd.ExcelWriter(tmp_path, engine='openpyxl')
    df = pd.DataFrame({'A': [1, 2, 3]})
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    print("First close() call:")
    writer.close()
    print("First close() succeeded")

    print("\nSecond close() call:")
    writer.close()  # This should raise an error
    print("Second close() succeeded")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)