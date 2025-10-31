import pytest
from hypothesis import given, strategies as st
import pandas.errors as pd_errors


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_swapped_values_bug(invalid_methodtype):
    """
    BUG: AbstractMethodError has swapped values in its error message.

    The error message currently says:
        "methodtype must be one of <invalid_value>, got <valid_types> instead."

    But it should say:
        "methodtype must be one of <valid_types>, got <invalid_value> instead."
    """
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    if "must be one of" in error_msg and "got" in error_msg:
        parts = error_msg.split("got")
        first_part = parts[0]
        second_part = parts[1] if len(parts) > 1 else ""

        valid_types_str = "{'method', 'classmethod', 'staticmethod', 'property'}"

        assert valid_types_str in first_part or all(t in first_part for t in ["method", "classmethod"]), \
            f"Valid types should appear in 'must be one of' clause, but got: {error_msg}"

        assert invalid_methodtype in second_part, \
            f"Invalid value should appear after 'got', but got: {error_msg}"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])