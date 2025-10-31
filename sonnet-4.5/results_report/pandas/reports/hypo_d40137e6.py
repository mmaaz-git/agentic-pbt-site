from hypothesis import given, strategies as st
import pandas as pd
import tempfile
import os

def test_close_idempotent():
    """Calling close() multiple times should not raise errors"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = pd.ExcelWriter(tmp_path, engine='openpyxl')
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()
        writer.close()  # Should not raise
        print("Test passed: close() is idempotent")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    test_close_idempotent()