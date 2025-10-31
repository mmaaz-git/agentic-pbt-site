import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

# Test with surrogate character as reported
print("Testing with surrogate character '\\ud800'...")
df = pd.DataFrame({'col': ['\ud800']})
print(f"DataFrame created: {df}")
print(f"DataFrame dtypes: {df.dtypes}")

try:
    interchange_obj = df.__dataframe__()
    print("Interchange object created successfully")
    result = from_dataframe(interchange_obj)
    print(f"Result: {result}")
    print("SUCCESS: No error occurred")
except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError occurred: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other error occurred: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()