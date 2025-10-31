from hypothesis import given, strategies as st
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

@given(st.dictionaries(st.text(min_size=1), st.integers().filter(lambda x: x < -100 or x > 100), min_size=1))
def test_error_message_clarity(logit_bias):
    try:
        validate_logit_bias(logit_bias)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between -100 and 100" in str(e), f"Expected specific error about range, got: {e}"

if __name__ == "__main__":
    test_error_message_clarity()