from hypothesis import given, strategies as st
import attrs
from attrs import converters

@given(st.floats().filter(lambda x: x not in [1.0, 0.0]))
def test_to_bool_rejects_undocumented_floats(x):
    try:
        converters.to_bool(x)
        assert False, f"to_bool should reject float {x}"
    except ValueError:
        pass

@given(st.sampled_from([1.0, 0.0]))
def test_to_bool_accepts_some_floats(x):
    result = converters.to_bool(x)
    assert result == (x == 1.0)

if __name__ == "__main__":
    print("Testing that to_bool rejects most floats but accepts 1.0 and 0.0...")
    print()

    print("Test 1: to_bool should reject floats other than 1.0 and 0.0")
    test_to_bool_rejects_undocumented_floats()
    print("✓ Passed - all floats except 1.0 and 0.0 are rejected")
    print()

    print("Test 2: to_bool accepts 1.0 and 0.0 (undocumented behavior)")
    test_to_bool_accepts_some_floats()
    print("✓ Passed - 1.0 converts to True and 0.0 converts to False")
    print()

    print("This demonstrates the bug: 1.0 and 0.0 are accepted despite not being")
    print("in the documented list of accepted values.")