import scipy.constants

key = 'proton mass'
val, unit, uncertainty = scipy.constants.physical_constants[key]

result = scipy.constants.precision(key)

print(f"Documentation claims: 5.1e-37")
print(f"Actual result: {result}")
print(f"Match: {result == 5.1e-37}")
print()
print(f"Additional details:")
print(f"  Value: {val}")
print(f"  Unit: {unit}")
print(f"  Uncertainty: {uncertainty}")
print(f"  Uncertainty/Value: {uncertainty/val}")
print(f"  Result matches uncertainty/value: {result == uncertainty/val}")