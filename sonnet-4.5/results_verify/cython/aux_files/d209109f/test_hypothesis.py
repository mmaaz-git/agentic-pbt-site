import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Distutils import Extension

@given(
    st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3),
    st.dictionaries(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        st.booleans(),
        min_size=1, max_size=3
    )
)
def test_extension_init_preserves_cython_params_with_pyrex(module_name, include_dirs, directives):
    ext = Extension(
        module_name,
        [f"{module_name}.pyx"],
        pyrex_gdb=True,
        cython_include_dirs=include_dirs,
        cython_directives=directives
    )

    assert ext.cython_include_dirs == include_dirs
    assert ext.cython_directives == directives

# Test with the specific failing inputs from the bug report - direct call
def test_specific():
    module_name = 'A'
    include_dirs = ['0']
    directives = {'A': False}

    ext = Extension(
        module_name,
        [f"{module_name}.pyx"],
        pyrex_gdb=True,
        cython_include_dirs=include_dirs,
        cython_directives=directives
    )

    print(f"Expected cython_include_dirs: {include_dirs}")
    print(f"Actual cython_include_dirs: {ext.cython_include_dirs}")
    print(f"Expected cython_directives: {directives}")
    print(f"Actual cython_directives: {ext.cython_directives}")

    assert ext.cython_include_dirs == include_dirs, f"Expected {include_dirs}, got {ext.cython_include_dirs}"
    assert ext.cython_directives == directives, f"Expected {directives}, got {ext.cython_directives}"

test_specific()