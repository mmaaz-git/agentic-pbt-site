from hypothesis import given, strategies as st
import django.template

@given(st.integers(min_value=0, max_value=1000))
def test_integer_trailing_period_property(num):
    text = f"{num}."
    var = django.template.Variable(text)

    if var.literal is not None and var.lookups is not None:
        assert False, f"Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'"

if __name__ == "__main__":
    test_integer_trailing_period_property()