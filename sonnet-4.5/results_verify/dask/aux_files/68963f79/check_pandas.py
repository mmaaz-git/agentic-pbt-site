import pandas as pd
import inspect

# Create a sample DataFrame
df = pd.DataFrame({'x': range(100)})

print("PANDAS HEAD() SIGNATURE:")
print(inspect.signature(df.head))
print("\nPANDAS HEAD() DOCUMENTATION:")
print(df.head.__doc__)

print("\n" + "="*60 + "\n")

print("PANDAS TAIL() SIGNATURE:")
print(inspect.signature(df.tail))
print("\nPANDAS TAIL() DOCUMENTATION:")
print(df.tail.__doc__)

print("\n" + "="*60 + "\n")

# Test that they both work the same way
print("Testing pandas methods:")
print(f"head(10) returns {len(df.head(10))} rows")
print(f"tail(10) returns {len(df.tail(10))} rows")