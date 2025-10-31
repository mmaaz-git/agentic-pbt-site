from hypothesis import given, strategies as st, settings
import pandas.errors

@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@settings(max_examples=10)
def test_abstract_method_error_invalid_methodtype_message_clarity(invalid_methodtype):
    try:
        pandas.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        print(f"Testing with invalid_methodtype='{invalid_methodtype}'")
        print(f"Error message: {error_msg}")

        valid_types = {"method", "classmethod", "staticmethod", "property"}
        parts = error_msg.split(", got")

        if len(parts) == 2:
            for valid_type in valid_types:
                if valid_type in parts[1]:
                    print(f"FAIL: Valid type '{valid_type}' appears in 'got X' part")
                    assert False, f"Valid type '{valid_type}' should not appear in 'got X' part"

        print("Test passed!")
        print("-" * 50)

# Run the test
test_abstract_method_error_invalid_methodtype_message_clarity()