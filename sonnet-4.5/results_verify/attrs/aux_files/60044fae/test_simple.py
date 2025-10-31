from attrs import converters

# Test the simple reproduction case
result_0 = converters.to_bool(0.0)
result_1 = converters.to_bool(1.0)

print(f"to_bool(0.0) = {result_0}")
print(f"to_bool(1.0) = {result_1}")

# Also test that integers work as expected
print(f"to_bool(0) = {converters.to_bool(0)}")
print(f"to_bool(1) = {converters.to_bool(1)}")

# Test other floats to see if they raise ValueError
test_values = [0.5, 1.5, 2.0, -1.0]
for val in test_values:
    try:
        result = converters.to_bool(val)
        print(f"to_bool({val}) = {result} (expected ValueError)")
    except ValueError as e:
        print(f"to_bool({val}) raised ValueError: {e}")