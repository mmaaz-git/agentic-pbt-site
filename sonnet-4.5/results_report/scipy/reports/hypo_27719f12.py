from hypothesis import given, strategies as st, assume, settings
from scipy.io.arff._arffread import DateAttribute

@given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20))
@settings(max_examples=100)
def test_date_format_invalid_patterns_should_raise(pattern):
    valid_components = ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']
    assume(not any(comp in pattern for comp in valid_components))
    assume('z' not in pattern.lower() and 'Z' not in pattern)

    try:
        result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")

        print(f"FAILED: Pattern '{pattern}' has no valid date components but returned "
              f"result='{result_pattern}', unit='{unit}' instead of raising ValueError. "
              f"This is due to bug on line 276: 'elif \"yy\":' which is always True")
        return False
    except ValueError:
        # This is the expected behavior
        return True

# Run the test
test_date_format_invalid_patterns_should_raise()