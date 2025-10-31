import pandas as pd
import numpy as np
import io

df = pd.DataFrame({'value': [np.finfo(np.float64).max]})

print("Original value:", df['value'].iloc[0])
print("Is finite?", np.isfinite(df['value'].iloc[0]))

json_buffer = io.StringIO()
df.to_json(json_buffer, orient='records')
json_buffer.seek(0)

# Let's also check what the JSON looks like
json_content = json_buffer.read()
print("\nJSON content:", json_content)
json_buffer.seek(0)

result = pd.read_json(json_buffer, orient='records')

print("\nAfter JSON round-trip:", result['value'].iloc[0])
print("Is finite?", np.isfinite(result['value'].iloc[0]))