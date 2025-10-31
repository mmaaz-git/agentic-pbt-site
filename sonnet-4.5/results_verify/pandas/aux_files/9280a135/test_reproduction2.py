import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

# Test with surrogate character as reported
print("Testing with surrogate character U+D800...")
df = pd.DataFrame({'col': ['\ud800']})
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes: {df.dtypes}")

try:
    interchange_obj = df.__dataframe__()
    print("Interchange object created successfully")
    result = from_dataframe(interchange_obj)
    print(f"Result shape: {result.shape}")
    print("SUCCESS: Conversion completed")
    print(f"Columns match: {list(df.columns) == list(result.columns)}")
except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError occurred!")
    print(f"Error message: {str(e).encode('ascii', 'replace').decode('ascii')}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}")
    print(f"Error message: {str(e).encode('ascii', 'replace').decode('ascii')}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()