import pandas.tseries.frequencies
from hypothesis import given, strategies as st, settings, assume

OFFSET_STRINGS = [
    'D', 'B', 'C', 'W', 'M', 'Q', 'Y',
    'BQ', 'BA', 'BM', 'BH', 'BQE', 'BQS', 'BYE', 'BYS',
    'MS', 'ME', 'QS', 'QE', 'YS', 'YE',
    'h', 'min', 's', 'ms', 'us', 'ns',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
]

@given(offset_str=st.sampled_from(OFFSET_STRINGS))
@settings(max_examples=500)
def test_get_period_alias_idempotence(offset_str):
    first_alias = pandas.tseries.frequencies.get_period_alias(offset_str)
    assume(first_alias is not None)

    second_alias = pandas.tseries.frequencies.get_period_alias(first_alias)

    assert second_alias == first_alias, (
        f"get_period_alias('{offset_str}') = '{first_alias}', "
        f"but get_period_alias('{first_alias}') = '{second_alias}'. "
        f"Expected idempotence: f(f(x)) should equal f(x)"
    )

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_get_period_alias_idempotence()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")