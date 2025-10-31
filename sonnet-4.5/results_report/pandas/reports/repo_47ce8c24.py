import pandas as pd

# Reproducing the bug: IntegerArray with base=1 and negative exponent
base = pd.array([1], dtype="Int64")
exponent = pd.array([-1], dtype="Int64")

print("Computing: base ** exponent")
print(f"base = {base}")
print(f"exponent = {exponent}")

try:
    result = base ** exponent
    print(f"result = {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")