import pandas as pd
import numpy as np
import time

# Create a large DataFrame to test performance
np.random.seed(42)
n = 100000
df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, n),
    'float_col': np.random.randn(n),
    'str_col': ['string_' + str(i % 100) for i in range(n)],
    'bool_col': np.random.choice([True, False], n)
})

print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes:\n{df.dtypes}\n")

# Time split orient
start = time.perf_counter()
split_result = df.to_dict(orient='split')
split_time = time.perf_counter() - start
print(f"Split orient time: {split_time:.4f} seconds")

# Time tight orient
start = time.perf_counter()
tight_result = df.to_dict(orient='tight')
tight_time = time.perf_counter() - start
print(f"Tight orient time: {tight_time:.4f} seconds")

print(f"\nTime difference: {tight_time - split_time:.4f} seconds ({(tight_time / split_time - 1) * 100:.1f}% slower)")

# Verify the data is the same
print(f"\nData equality check: {split_result['data'] == tight_result['data']}")