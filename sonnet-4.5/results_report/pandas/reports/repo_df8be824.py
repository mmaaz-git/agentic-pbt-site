import pandas as pd
import io

# Test with value that's below int64 minimum (-2^63)
# int64 min is -9223372036854775808
# Testing with -9223372036854775809 (int64 min - 1)
df = pd.DataFrame({'col': [-9223372036854775809]})

print("Testing pandas JSON round-trip for large negative integer (-9223372036854775809)")
print("="*70)

for orient in ['split', 'records', 'index', 'columns', 'values', 'table']:
    print(f"\nTesting orient='{orient}':")
    try:
        # Attempt to convert to JSON
        json_str = df.to_json(orient=orient)
        print(f"  to_json: Success")
        print(f"  JSON output: {json_str[:100]}{'...' if len(json_str) > 100 else ''}")

        # Attempt to read back the JSON
        df_roundtrip = pd.read_json(io.StringIO(json_str), orient=orient)
        print(f"  read_json: Success")
        print(f"  Roundtrip value: {df_roundtrip['col'][0]}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")