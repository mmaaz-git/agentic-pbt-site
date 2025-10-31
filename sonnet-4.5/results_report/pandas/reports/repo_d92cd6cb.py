import pandas as pd
import pandas.api.interchange as interchange

# Create a DataFrame with a surrogate character in the string column
df = pd.DataFrame({'A': [0], 'B': [0.0], 'C': ['\ud800']})

print("Created DataFrame with surrogate character U+D800 in column C")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes:\n{df.dtypes}")

# Try to convert through the interchange protocol
try:
    print("\nAttempting to convert through interchange protocol...")
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    print("Success! Result:")
    print(f"Result shape: {result.shape}")
    print(f"Result dtypes:\n{result.dtypes}")
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()