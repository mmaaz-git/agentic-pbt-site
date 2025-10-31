from hypothesis import given, strategies as st
from Cython.Distutils import Extension


@given(
    pyrex_val=st.booleans(),
    cython_val=st.booleans()
)
def test_explicit_cython_params_not_overridden_by_pyrex(pyrex_val, cython_val):
    ext = Extension(
        "test",
        ["test.pyx"],
        **{f"pyrex_gdb": pyrex_val},
        cython_gdb=cython_val
    )

    assert ext.cython_gdb == cython_val, \
        f"Explicit cython_gdb={cython_val} was overridden by pyrex_gdb={pyrex_val}"


@given(
    pyrex_list=st.lists(st.text(min_size=1, max_size=20), max_size=3),
    cython_list=st.lists(st.text(min_size=1, max_size=20), max_size=3)
)
def test_explicit_cython_include_dirs_not_overridden_by_pyrex(pyrex_list, cython_list):
    ext = Extension(
        "test",
        ["test.pyx"],
        **{"pyrex_include_dirs": pyrex_list},
        cython_include_dirs=cython_list
    )

    assert ext.cython_include_dirs == cython_list


if __name__ == "__main__":
    # Run the tests
    test_explicit_cython_params_not_overridden_by_pyrex()
    test_explicit_cython_include_dirs_not_overridden_by_pyrex()