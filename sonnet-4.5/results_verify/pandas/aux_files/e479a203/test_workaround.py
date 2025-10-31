import pandas as pd
import numpy as np
import io

max_val = np.finfo(np.float64).max
df = pd.DataFrame({'value': [max_val]})

# Test different values near the max
test_values = [
    max_val,
    max_val * 0.99999,
    max_val * 0.9999,
    1e308,
    1.797693e308
]

for val in test_values:
    df = pd.DataFrame({'value': [val]})
    json_buffer = io.StringIO()
    df.to_json(json_buffer, orient='records')
    json_buffer.seek(0)

    json_content = json_buffer.read()
    json_buffer.seek(0)

    result = pd.read_json(json_buffer, orient='records')

    print(f"Original: {val:.10e}, JSON: {json_content[:30]}..., Result: {result['value'].iloc[0]}, Is finite: {np.isfinite(result['value'].iloc[0])}")