import scipy.constants

# Test with a few different constants to understand the pattern
test_keys = ['proton mass', 'electron mass', 'neutron mass', 'speed of light in vacuum']

for key in test_keys:
    try:
        val, unit, uncertainty = scipy.constants.physical_constants[key]
        result = scipy.constants.precision(key)

        print(f"\nConstant: {key}")
        print(f"  Value: {val} {unit}")
        print(f"  Uncertainty: {uncertainty} {unit}")
        print(f"  precision() returns: {result}")
        print(f"  uncertainty/value: {uncertainty/val}")
        print(f"  Match: {result == uncertainty/val}")
    except Exception as e:
        print(f"\nConstant: {key} - Error: {e}")