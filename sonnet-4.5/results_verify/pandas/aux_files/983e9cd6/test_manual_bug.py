import pandas as pd
from pandas.io.json import read_json, to_json
import io

print("Testing manual reproduction of the bug...")
print()

df = pd.DataFrame([{'b': 0.0}])
print(f"Original dtype: {df['b'].dtype}")

json_str = to_json(None, df, orient='records')
print(f"JSON string: {json_str}")

result = read_json(io.StringIO(json_str), orient='records')
print(f"Result dtype: {result['b'].dtype}")

print()
print("Attempting assertion...")
try:
    assert df['b'].dtype == result['b'].dtype
    print("Assertion passed (no bug)")
except AssertionError:
    print("AssertionError: dtypes do not match!")
    print(f"  Expected: {df['b'].dtype}")
    print(f"  Got: {result['b'].dtype}")

print()
print("Testing different orient values...")
for orient in ['records', 'split', 'columns', 'index', 'table']:
    try:
        df_test = pd.DataFrame([{'a': 1, 'b': 0.0}])
        json_str = to_json(None, df_test, orient=orient)
        result = read_json(io.StringIO(json_str), orient=orient)

        if df_test['b'].dtype == result['b'].dtype:
            print(f"✓ orient='{orient}': dtypes preserved")
        else:
            print(f"✗ orient='{orient}': dtype changed from {df_test['b'].dtype} to {result['b'].dtype}")
    except Exception as e:
        print(f"✗ orient='{orient}': Error - {e}")