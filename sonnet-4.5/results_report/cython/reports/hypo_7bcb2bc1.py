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
test_get_extension_attr_falsy_command_values()