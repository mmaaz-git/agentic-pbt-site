import pandas.api.typing as pat
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=1000))
def test_nattype_singleton_property(n):
    instances = [pat.NaTType() for _ in range(n) if n > 0]
    if instances:
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, f"NaTType() should be a singleton (like NAType), but got different instances"


if __name__ == "__main__":
    test_nattype_singleton_property()