from hypothesis import given, strategies as st, settings, assume
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options


@given(st.text(min_size=1), st.integers())
@settings(max_examples=1000)
def test_restore_removes_added_keys(new_attr_name, new_value):
    assume(new_attr_name.isidentifier())
    assume(not hasattr(Options, new_attr_name))

    backup = backup_Options()

    setattr(Options, new_attr_name, new_value)
    assert hasattr(Options, new_attr_name)

    restore_Options(backup)

    assert not hasattr(Options, new_attr_name), \
        f"restore_Options should remove new attribute {new_attr_name}"

if __name__ == "__main__":
    # Run with specific failing input first
    test_restore_removes_added_keys('A', 0)
    print("Test with specific input passed!")

    # Run property-based testing
    test_restore_removes_added_keys()
    print("All property-based tests passed!")