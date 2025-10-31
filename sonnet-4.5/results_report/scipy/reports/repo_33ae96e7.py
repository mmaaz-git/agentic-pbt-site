import scipy.constants as sc

# Test the precision function with a negative-valued physical constant
key = 'electron magn. moment'
result = sc.precision(key)
value_const, unit_const, abs_precision = sc.physical_constants[key]

print(f"Physical constant: {key}")
print(f"Value: {value_const}")
print(f"Unit: {unit_const}")
print(f"Absolute precision: {abs_precision}")
print(f"precision(key) returned: {result}")
print(f"Expected (using standard physics definition): {abs(abs_precision / value_const)}")
print()

# The standard physics definition of relative precision should always be positive
# as it represents the magnitude of uncertainty relative to the measured value
assert result > 0, f"Relative precision should be positive, but got {result}"