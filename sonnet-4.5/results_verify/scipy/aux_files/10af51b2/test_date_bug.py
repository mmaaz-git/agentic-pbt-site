from hypothesis import given, strategies as st, settings, assume
from scipy.io.arff._arffread import DateAttribute


@given(st.text().filter(lambda x: 'yyyy' not in x.lower() and 'yy' not in x.lower()
                                     and 'mm' not in x.lower() and 'dd' not in x.lower()
                                     and 'hh' not in x.lower() and 'ss' not in x.lower()))
@settings(max_examples=100)
def test_date_format_no_components_should_fail(text):
    assume(len(text.strip()) > 0)

    try:
        pattern, unit = DateAttribute._get_date_format(f"date {text}")
        print(f"FAIL: Should have raised ValueError for input 'date {text}', but got unit={unit}")
        assert False
    except ValueError:
        pass  # Expected behavior

# Run the test
test_date_format_no_components_should_fail()