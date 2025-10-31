from hypothesis import given, strategies as st
from django.core.checks import CheckMessage, Error, ERROR


@given(st.text())
def test_checkmessage_equality_symmetry_with_subclass(msg):
    parent = CheckMessage(ERROR, msg)
    child = Error(msg)

    assert (parent == child) == (child == parent), (
        f"Symmetry violated: CheckMessage == Error is {parent == child}, "
        f"but Error == CheckMessage is {child == parent}"
    )

# Run the test
if __name__ == "__main__":
    test_checkmessage_equality_symmetry_with_subclass()
    print("Test passed!")