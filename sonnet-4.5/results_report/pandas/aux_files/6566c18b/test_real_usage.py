import pandas as pd
import io

# Test real pandas usage with negative chunksize
csv_data = "a,b,c\n1,2,3\n4,5,6"

try:
    # This should raise ValueError but doesn't
    reader = pd.read_csv(io.StringIO(csv_data), chunksize=-1.0)
    print("SUCCESS: Reader created with chunksize=-1.0 (should have failed!)")

    # Try to read from it
    for i, chunk in enumerate(reader):
        print(f"Chunk {i}: shape={chunk.shape}")
        if i > 2:  # Safety limit
            break

except ValueError as e:
    print(f"EXPECTED: ValueError raised: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")