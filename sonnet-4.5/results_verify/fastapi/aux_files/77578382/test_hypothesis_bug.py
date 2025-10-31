from hypothesis import given, strategies as st
import attr

@attr.s
class Container:
    data = attr.ib()

@given(st.lists(st.integers(), min_size=1))
def test_value_serializer_inst_field_never_none(items):
    def serializer(inst, field, value):
        assert inst is not None, "inst should never be None per docs"
        assert field is not None, "field should never be None per docs"
        return value

    obj = Container(data=items)
    attr.asdict(obj, recurse=True, value_serializer=serializer)

# Run the test
if __name__ == "__main__":
    # Hypothesis decorates the function, so we need to call it properly
    try:
        test_value_serializer_inst_field_never_none()
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")