from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

VALID_FREQS = ["D", "B", "C", "M", "h", "min", "s", "ms", "us", "ns", "W", "Y", "Q"]

@given(
    source=st.sampled_from(VALID_FREQS),
    target=st.sampled_from(VALID_FREQS)
)
def test_subperiod_superperiod_inverse(source, target):
    """Test that is_subperiod and is_superperiod maintain inverse relationship.

    The mathematical property being tested:
    is_subperiod(A, B) == is_superperiod(B, A) for all valid frequency pairs

    This should always hold because:
    - is_subperiod(A, B) checks if you can downsample from A to B
    - is_superperiod(B, A) checks if you can upsample from B to A
    - These operations are mathematical inverses of each other
    """
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)

    assert result_sub == result_super, (
        f"Inverse relationship violated: "
        f"is_subperiod({source!r}, {target!r}) = {result_sub} but "
        f"is_superperiod({target!r}, {source!r}) = {result_super}"
    )

# Run the test
if __name__ == "__main__":
    test_subperiod_superperiod_inverse()