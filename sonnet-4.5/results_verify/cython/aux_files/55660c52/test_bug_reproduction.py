import json

def validate_logit_bias(logit_bias):
    if logit_bias is None:
        return None

    if isinstance(logit_bias, str):
        logit_bias = json.loads(logit_bias)

    validated_logit_bias = {}
    for key, value in logit_bias.items():
        try:
            int_key = int(key)
            int_value = int(value)
            if -100 <= int_value <= 100:
                validated_logit_bias[int_key] = int_value
            else:
                raise ValueError("Value must be between -100 and 100")
        except ValueError:
            raise ValueError("Invalid key-value pair in logit_bias dictionary")

    return validated_logit_bias

# Test the specific example
print("Testing with {'100': 150}:")
try:
    validate_logit_bias({"100": 150})
except ValueError as e:
    print(f"Error message: {e}")

# Test with other out-of-range values
print("\nTesting with {'100': -101}:")
try:
    validate_logit_bias({"100": -101})
except ValueError as e:
    print(f"Error message: {e}")

# Test with invalid int conversion
print("\nTesting with {'not_a_number': 50}:")
try:
    validate_logit_bias({"not_a_number": 50})
except ValueError as e:
    print(f"Error message: {e}")

# Test with value that can't be converted to int
print("\nTesting with {'100': 'not_a_number'}:")
try:
    validate_logit_bias({"100": "not_a_number"})
except ValueError as e:
    print(f"Error message: {e}")