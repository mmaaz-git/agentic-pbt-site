import pandas as pd
from pandas.io.json import read_json, to_json
import io

print("Testing orient='table' specifically...")

df_test = pd.DataFrame([{'a': 1, 'b': 0.0}])
print(f"Original dtype for column 'b': {df_test['b'].dtype}")

# For table orient, we need to use date_format='iso' if there are dates
json_str = to_json(None, df_test, orient='table', date_format='iso')
print(f"JSON string (table format): {json_str[:200]}...")  # Print first 200 chars

result = read_json(io.StringIO(json_str), orient='table')
print(f"Result dtype for column 'b': {result['b'].dtype}")

if df_test['b'].dtype == result['b'].dtype:
    print("✓ orient='table': dtypes preserved!")
else:
    print(f"✗ orient='table': dtype changed from {df_test['b'].dtype} to {result['b'].dtype}")