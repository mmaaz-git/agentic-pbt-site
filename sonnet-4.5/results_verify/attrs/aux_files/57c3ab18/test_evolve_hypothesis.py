from hypothesis import given, strategies as st
import attr
import attrs

@given(st.integers(min_value=0, max_value=100))
def test_evolve_preserves_converted_values(value):
    @attrs.define
    class MyClass:
        x: int = attr.field(converter=lambda v: v * 2)

    original = MyClass(x=value)
    assert original.x == value * 2

    evolved = attr.evolve(original)
    assert evolved.x == original.x

# Run the test
if __name__ == "__main__":
    test_evolve_preserves_converted_values()