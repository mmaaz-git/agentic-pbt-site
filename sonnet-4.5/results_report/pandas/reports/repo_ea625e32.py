import pandas as pd
from io import StringIO

# Create a Series with float64 dtype containing whole numbers
series = pd.Series([1.0, 2.0, 3.0])
print(f"Original Series:\n{series}")
print(f"Original dtype: {series.dtype}")
print()

# Serialize to JSON
json_str = series.to_json(orient='split')
print(f"JSON representation:\n{json_str}")
print()

# Read back from JSON
recovered = pd.read_json(StringIO(json_str), orient='split', typ='series')
print(f"Recovered Series:\n{recovered}")
print(f"Recovered dtype: {recovered.dtype}")
print()

# Check if dtypes match
print(f"Dtypes match: {series.dtype == recovered.dtype}")
print(f"Values equal: {(series.values == recovered.values).all()}")

# Show that this affects calculations
print("\n--- Impact on calculations ---")
print(f"Original series / 2:\n{series / 2}")
print(f"Recovered series / 2:\n{recovered / 2}")