from hypothesis import given, strategies as st
from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

@given(
    cmd_value=st.one_of(
        st.just(0),
        st.just(False),
        st.just([]),
        st.just(""),
    ),
    ext_value=st.one_of(
        st.just(1),
        st.just(True),
        st.just(["/some/path"]),
        st.just("something"),
    ),
)
def test_get_extension_attr_falsy_command_values(cmd_value, ext_value):
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()
    cmd.finalize_options()

    cmd.cython_gdb = cmd_value

    ext = Extension("test_module", ["test.pyx"])
    ext.cython_gdb = ext_value

    result = cmd.get_extension_attr(ext, 'cython_gdb')

    assert result == cmd_value, \
        f"Expected get_extension_attr to return command value {cmd_value!r}, but got {result!r}"

# Run the test
if __name__ == "__main__":
    # Define a simple function without the hypothesis decorator for manual testing
    def manual_test(cmd_value, ext_value):
        dist = Distribution()
        cmd = build_ext(dist)
        cmd.initialize_options()
        cmd.finalize_options()

        cmd.cython_gdb = cmd_value

        ext = Extension("test_module", ["test.pyx"])
        ext.cython_gdb = ext_value

        result = cmd.get_extension_attr(ext, 'cython_gdb')

        assert result == cmd_value, \
            f"Expected get_extension_attr to return command value {cmd_value!r}, but got {result!r}"

    # Test with specific failing input mentioned in the bug report
    print("Testing with specific failing input: cmd_value=0, ext_value=1")
    try:
        manual_test(0, 1)
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nTesting with cmd_value=False, ext_value=True")
    try:
        manual_test(False, True)
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nTesting with cmd_value=[], ext_value=['/some/path']")
    try:
        manual_test([], ['/some/path'])
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nTesting with cmd_value='', ext_value='something'")
    try:
        manual_test('', 'something')
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")