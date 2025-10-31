import scipy.constants as sc

# Test the bug as reported
key = 'electron magn. moment'
result = sc.precision(key)
value_const, unit_const, abs_precision = sc.physical_constants[key]

print(f"Value: {value_const}")
print(f"Absolute precision: {abs_precision}")
print(f"precision(key) returned: {result}")
print(f"Expected (according to bug report): {abs(abs_precision / value_const)}")
print(f"Is result negative?: {result < 0}")

# Check other constants with negative values
print("\n--- Testing other constants ---")
for test_key in sc.physical_constants.keys():
    value, unit, uncertainty = sc.physical_constants[test_key]
    if value < 0 and uncertainty != 0:
        prec = sc.precision(test_key)
        print(f"{test_key}: value={value:.3e}, precision={prec:.3e}, negative?={prec < 0}")

# Test positive value constant for comparison
print("\n--- Testing positive constant for comparison ---")
key = 'proton mass'
result = sc.precision(key)
value_const, unit_const, abs_precision = sc.physical_constants[key]
print(f"Key: {key}")
print(f"Value: {value_const}")
print(f"Absolute precision: {abs_precision}")
print(f"precision(key) returned: {result}")
print(f"Is result negative?: {result < 0}")