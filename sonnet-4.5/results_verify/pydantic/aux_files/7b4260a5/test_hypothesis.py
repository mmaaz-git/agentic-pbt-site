from hypothesis import given, strategies as st, settings
from pydantic.alias_generators import to_pascal


@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_to_pascal_idempotent(field_name):
    """to_pascal applied twice should equal to_pascal applied once (idempotence)."""
    once = to_pascal(field_name)
    twice = to_pascal(once)
    assert once == twice, f"Failed for input '{field_name}': to_pascal('{field_name}')='{once}', to_pascal('{once}')='{twice}'"

if __name__ == "__main__":
    try:
        test_to_pascal_idempotent()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")