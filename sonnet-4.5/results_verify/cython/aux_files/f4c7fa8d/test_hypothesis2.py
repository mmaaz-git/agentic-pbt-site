from hypothesis import given, strategies as st, settings, assume
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

def test_with_specific_input():
    """Test with the specific failing input from the bug report"""
    print("Testing with specific input: new_attr_name='A', new_value=0")

    # Check preconditions
    if hasattr(Options, 'A'):
        delattr(Options, 'A')

    backup = backup_Options()
    setattr(Options, 'A', 0)
    assert hasattr(Options, 'A')

    try:
        restore_Options(backup)
        assert not hasattr(Options, 'A'), \
            f"restore_Options should remove new attribute 'A'"
        print("Test passed - attribute was removed")
    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
        return False
    return True

@given(st.text(min_size=1), st.integers())
@settings(max_examples=100)  # Reduced for faster testing
def test_restore_removes_added_keys(new_attr_name, new_value):
    assume(new_attr_name.isidentifier())
    assume(not hasattr(Options, new_attr_name))

    backup = backup_Options()

    setattr(Options, new_attr_name, new_value)
    assert hasattr(Options, new_attr_name)

    try:
        restore_Options(backup)
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            raise AssertionError(f"RuntimeError during restore_Options: {e}")
        raise

    assert not hasattr(Options, new_attr_name), \
        f"restore_Options should remove new attribute {new_attr_name}"

if __name__ == "__main__":
    # Test with specific input
    success = test_with_specific_input()
    if not success:
        print("Bug confirmed with specific input!")

    # Run property-based testing
    try:
        test_restore_removes_added_keys()
        print("All property-based tests passed!")
    except Exception as e:
        print(f"Property-based testing failed: {e}")