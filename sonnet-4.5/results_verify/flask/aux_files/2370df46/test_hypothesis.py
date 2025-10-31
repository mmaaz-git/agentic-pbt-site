import pandas as pd
from pandas.api.typing import NaTType
from hypothesis import given, strategies as st


@given(st.integers(min_value=0, max_value=100))
def test_nattype_singleton_property(n):
    instances = [NaTType() for _ in range(n)]
    if len(instances) > 0:
        for instance in instances:
            assert instance is pd.NaT, f"NaTType() should return the singleton pd.NaT"


def test_nattype_constructors_return_same_instance():
    nat1 = NaTType()
    nat2 = NaTType()
    assert nat1 is nat2, "Multiple NaTType() calls should return the same singleton instance"

if __name__ == "__main__":
    # Run the tests manually without hypothesis wrapper
    def test_manually():
        n = 1
        instances = [NaTType() for _ in range(n)]
        if len(instances) > 0:
            for instance in instances:
                assert instance is pd.NaT, f"NaTType() should return the singleton pd.NaT"

    try:
        test_manually()
        print("Manual singleton test passed")
    except AssertionError as e:
        print(f"Manual singleton test failed: {e}")

    try:
        test_nattype_constructors_return_same_instance()
        print("Constructor test passed")
    except AssertionError as e:
        print(f"Constructor test failed: {e}")