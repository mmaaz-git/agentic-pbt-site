import numpy as np

print("Testing numpy.around with various decimal values:")
print("-" * 50)

test_value = 2.2250738585072014e-308

# Test with increasing decimal values
test_decimals = [10, 50, 100, 200, 300, 310, 320, 400, 500]

for decimals in test_decimals:
    try:
        result = np.around(test_value, decimals)
        print(f"np.around({test_value:.3e}, {decimals:3d}) = {result:20} | isnan={np.isnan(result)}")
    except Exception as e:
        print(f"np.around({test_value:.3e}, {decimals:3d}) raised: {e}")

print("\n" + "-" * 50)
print("Testing with other small values:")
other_values = [1e-100, 1e-200, 1e-300, 1e-307, 1e-308]
for val in other_values:
    # Calculate what pandas would compute for digits
    frac, whole = np.modf(val)
    if whole == 0 and frac != 0:
        digits = -int(np.floor(np.log10(abs(frac)))) - 1 + 3  # precision=3
    else:
        digits = 3
    result = np.around(val, digits)
    print(f"Value: {val:.3e}, Computed digits: {digits:3d}, Result: {result}, isnan={np.isnan(result)}")

print("\n" + "-" * 50)
print("Documentation check - what's the maximum safe decimals for numpy.around?")
print("Testing boundary...")
for decimals in [290, 295, 300, 305, 308, 309, 310]:
    result = np.around(1e-305, decimals)
    print(f"decimals={decimals}: {'NaN' if np.isnan(result) else 'OK'}")