import pandas.tseries.frequencies as freq_module
from hypothesis import given, strategies as st, settings, example

PERIOD_FREQUENCIES = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'B', 'BM', 'BQ', 'BY']

@given(
    source=st.sampled_from(PERIOD_FREQUENCIES),
    target=st.sampled_from(PERIOD_FREQUENCIES),
)
@example(source='D', target='B')
@settings(max_examples=200)
def test_is_subperiod_superperiod_symmetry(source, target):
    is_sub = freq_module.is_subperiod(source, target)
    is_super = freq_module.is_superperiod(target, source)

    assert is_sub == is_super, (
        f"Symmetry violated: is_subperiod({source}, {target})={is_sub} but "
        f"is_superperiod({target}, {source})={is_super}"
    )

if __name__ == "__main__":
    test_is_subperiod_superperiod_symmetry()