import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQ_STRINGS = [
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'ME', 'QE', 'YE',
    'B', 'BME', 'BQE', 'BYE',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
    'MS', 'QS', 'YS', 'BMS', 'BQS', 'BYS',
]

@given(
    source=st.sampled_from(VALID_FREQ_STRINGS),
    target=st.sampled_from(VALID_FREQ_STRINGS)
)
@settings(max_examples=500)
def test_subperiod_superperiod_symmetry(source, target):
    try:
        is_sub = freq.is_subperiod(source, target)
        is_super = freq.is_superperiod(target, source)

        assert is_sub == is_super, (
            f"Symmetry broken: is_subperiod({source}, {target}) = {is_sub}, "
            f"but is_superperiod({target}, {source}) = {is_super}"
        )
    except (ValueError, KeyError) as e:
        pass

# Run the test
if __name__ == "__main__":
    try:
        test_subperiod_superperiod_symmetry()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")