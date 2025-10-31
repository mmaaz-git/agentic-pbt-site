from hypothesis import given, strategies as st, example
import pandas as pd
import pytest


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@example("invalid_type")  # Explicitly provide the failing example
def test_abstract_method_error_validation_message(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    # The bug: The error message has swapped parameters
    # It says "methodtype must be one of invalid_type, got {'method', ...} instead"
    # When it should say "methodtype must be one of {'method', ...}, got invalid_type instead"

    # This assertion will FAIL due to the bug
    # We expect the valid types to appear after "must be one of"
    if "must be one of" in error_message:
        parts = error_message.split("must be one of")[1].split(",")[0].strip()
        # The bug causes parts to be the invalid_methodtype, not the valid types
        print(f"Test failed for input '{invalid_methodtype}'")
        print(f"Error message: {error_message}")
        print(f"After 'must be one of': '{parts}'")
        print(f"Expected to see valid types but saw: '{parts}'")

        # This assertion demonstrates the bug
        assert parts != invalid_methodtype, f"Bug confirmed: Error message has swapped parameters. The invalid input '{invalid_methodtype}' appears where valid options should be."


if __name__ == "__main__":
    test_abstract_method_error_validation_message()