from hypothesis import given, strategies as st, settings
from Cython.Utility.Dataclasses import field


@given(st.booleans() | st.none())
@settings(max_examples=100)
def test_field_repr_contains_kw_only(kw_only):
    result = field(kw_only=kw_only, default=42)
    repr_str = repr(result)

    assert "kw_only=" in repr_str, (
        f"Expected 'kw_only=' in repr output, but got:\n{repr_str}"
    )

# Run the test
if __name__ == "__main__":
    test_field_repr_contains_kw_only()