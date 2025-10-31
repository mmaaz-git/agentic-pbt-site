import pandas as pd
import numpy as np
import io

max_val = np.finfo(np.float64).max
df = pd.DataFrame({'value': [max_val]})

print("Original value:", df['value'].iloc[0])
print("Original value == max float64:", df['value'].iloc[0] == max_val)

# Test with different double_precision values
for precision in [10, 15, 17]:
    try:
        json_buffer = io.StringIO()
        df.to_json(json_buffer, orient='records', double_precision=precision)
        json_buffer.seek(0)

        json_content = json_buffer.read()
        print(f"\nPrecision {precision}:")
        print(f"  JSON: {json_content}")
        json_buffer.seek(0)

        # Try with precise_float=True
        result = pd.read_json(json_buffer, orient='records', precise_float=True)
        print(f"  After round-trip (precise_float=True): {result['value'].iloc[0]}")
        print(f"  Is finite?: {np.isfinite(result['value'].iloc[0])}")

        # Try with precise_float=False (default)
        json_buffer.seek(0)
        result = pd.read_json(json_buffer, orient='records', precise_float=False)
        print(f"  After round-trip (precise_float=False): {result['value'].iloc[0]}")
        print(f"  Is finite?: {np.isfinite(result['value'].iloc[0])}")

    except ValueError as e:
        print(f"\nPrecision {precision}: ERROR - {e}")