import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe
import traceback

# Create a DataFrame with a Unicode surrogate character
df = pd.DataFrame({'col': ['\ud800']})
print("Original DataFrame created")
print(f"Shape: {df.shape}")
print(f"String value representation: {repr(df['col'][0])}")

# Try to convert through interchange protocol
try:
    interchange_obj = df.__dataframe__()
    print("Interchange object created successfully")
    df_roundtrip = from_dataframe(interchange_obj)
    print("Roundtrip succeeded:")
    print(f"Roundtrip shape: {df_roundtrip.shape}")
except Exception as e:
    print(f"\nError during roundtrip conversion:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()