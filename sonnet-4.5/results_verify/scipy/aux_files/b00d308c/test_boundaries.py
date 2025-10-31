import scipy.stats
import numpy as np
import sys

print(f"System float info:")
print(f"Min positive float: {sys.float_info.min}")
print(f"Epsilon: {sys.float_info.epsilon}")

# Test various p values around the problematic range
test_values = [
    1e-308,
    5e-309,
    2.225e-308,  # Close to min normal float
    1e-309,
    5e-324,  # Smallest denormalized float
]

for p_val in test_values:
    print(f"\nTesting p={p_val}")
    for n in [1, 2, 3, 4, 10]:
        try:
            result = scipy.stats.binom.pmf(0, n, p_val)
            print(f"  n={n}: pmf(0)={result}")
        except OverflowError as e:
            print(f"  n={n}: OverflowError")
        except Exception as e:
            print(f"  n={n}: {type(e).__name__}: {e}")