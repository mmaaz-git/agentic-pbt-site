import pandas as pd
import tempfile
import os

print("Testing double close() on CSVWriter...")
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df = pd.DataFrame({'A': [1, 2, 3]})
    # CSV doesn't have a writer with close() method like Excel does
    # Let's test with regular file writing
    with open(tmp_path, 'w') as f:
        df.to_csv(f)

    # Try reopening and closing twice
    f = open(tmp_path, 'w')
    df.to_csv(f)
    f.close()
    f.close()  # Should not raise
    print("CSV file double close() succeeded")
except Exception as e:
    print(f"CSV file error: {e}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

# Test with HDFStore which also has a close() method
print("\nTesting double close() on HDFStore...")
with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df = pd.DataFrame({'A': [1, 2, 3]})
    store = pd.HDFStore(tmp_path)
    store['df'] = df
    store.close()
    store.close()  # Let's see what happens
    print("HDFStore double close() succeeded")
except Exception as e:
    print(f"HDFStore error on double close(): {type(e).__name__}: {e}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)