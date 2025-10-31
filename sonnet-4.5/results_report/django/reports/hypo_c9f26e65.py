from hypothesis import given, strategies as st
from django.template import Variable

@given(
    st.integers(),
    st.booleans()
)
def test_variable_literal_trailing_dot_bug(num, add_trailing_dot):
    var_string = str(num)
    if add_trailing_dot:
        var_string = var_string + '.'

    try:
        v = Variable(var_string)

        if v.literal is not None and isinstance(v.literal, float):
            assert var_string[-1] != '.', f"Variable with trailing dot '{var_string}' should not parse as valid float but got {v.literal}"
    except ValueError:
        pass

# Run the test
if __name__ == "__main__":
    test_variable_literal_trailing_dot_bug()