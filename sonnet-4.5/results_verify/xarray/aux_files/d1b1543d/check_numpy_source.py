import numpy as np
import warnings

# Check NumPy's behavior in detail
print("NumPy cov behavior with edge cases:")

# Test with increasing ddof values
test_data = [1.0, 2.0, 3.0]

for ddof in [0, 1, 2, 3, 4, 5]:
    print(f"\nddof={ddof}, data length={len(test_data)}:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = np.cov(test_data, ddof=ddof)
        print(f"  Result: {result}")
        if w:
            for warning in w:
                print(f"  Warning: {warning.message}")

        # Check the calculation manually
        mean = np.mean(test_data)
        demeaned = test_data - mean
        sum_sq = np.sum(demeaned**2)
        n = len(test_data)

        if n - ddof > 0:
            manual_cov = sum_sq / (n - ddof)
            print(f"  Manual calculation: {manual_cov}")
        else:
            print(f"  Manual calculation: Division by zero or negative (n-ddof={n-ddof})")