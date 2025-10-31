import numpy as np
import matplotlib.pyplot as plt

test_values = [
    ("Small single value", [1.0]),
    ("Zero single value", [0.0]),
    ("Large single value", [1e16]),
    ("Negative large single value", [-1e16]),
    ("Multiple same values", [1e16, 1e16, 1e16]),
]

for name, values in test_values:
    print(f"\n{name}: {values}")
    arr = np.array(values)
    print(f"  Min: {arr.min()}, Max: {arr.max()}, Range: {arr.max() - arr.min()}")

    try:
        # Try numpy histogram directly
        hist, bins = np.histogram(arr)
        print(f"  ✓ numpy.histogram works - bins: {len(bins)-1}")
    except Exception as e:
        print(f"  ✗ numpy.histogram failed: {e}")

    try:
        # Try matplotlib hist
        plt.figure()
        plt.hist(arr)
        plt.close()
        print(f"  ✓ plt.hist works")
    except Exception as e:
        print(f"  ✗ plt.hist failed: {e}")