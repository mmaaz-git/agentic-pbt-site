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

# Run the test with the specific failing input
print("Testing with specific failing input: new_attr_name='A', new_value=0")
try:
    # Manually test the case rather than invoking the decorated function
    new_attr_name = 'A'
    new_value = 0

    if new_attr_name.isidentifier() and not hasattr(Options, new_attr_name):
        backup = backup_Options()
        setattr(Options, new_attr_name, new_value)
        assert hasattr(Options, new_attr_name)

        restore_Options(backup)

        assert not hasattr(Options, new_attr_name), \
            f"restore_Options should remove new attribute {new_attr_name}"
        print("Test passed!")
except Exception as e:
    print(f"Test failed with error: {type(e).__name__}: {e}")