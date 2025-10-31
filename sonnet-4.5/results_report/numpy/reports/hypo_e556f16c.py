from hypothesis import given, strategies as st
import numpy.rec

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_format_parser_rejects_integer_names(int_names):
    formats = ['i4'] * len(int_names)
    try:
        numpy.rec.format_parser(formats, int_names, [])
        assert False, f"Should have raised error for integer names {int_names}"
    except (TypeError, ValueError):
        pass
    except AttributeError:
        raise AssertionError("Got unhelpful AttributeError instead of clear TypeError/ValueError")

# Run the test
test_format_parser_rejects_integer_names()