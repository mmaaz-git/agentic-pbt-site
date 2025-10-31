import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

# Test read_clipboard with non-UTF-8 encoding
try:
    read_clipboard(encoding='latin-1')
except Exception as e:
    print(f"read_clipboard with 'latin-1': {type(e).__name__}: {e}")

# Test to_clipboard with non-UTF-8 encoding
try:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    to_clipboard(df, encoding='latin-1')
except Exception as e:
    print(f"to_clipboard with 'latin-1': {type(e).__name__}: {e}")

# Test with another non-UTF-8 encoding
try:
    read_clipboard(encoding='iso-8859-1')
except Exception as e:
    print(f"read_clipboard with 'iso-8859-1': {type(e).__name__}: {e}")

try:
    df = pd.DataFrame([[5, 6]], columns=['X', 'Y'])
    to_clipboard(df, encoding='iso-8859-1')
except Exception as e:
    print(f"to_clipboard with 'iso-8859-1': {type(e).__name__}: {e}")