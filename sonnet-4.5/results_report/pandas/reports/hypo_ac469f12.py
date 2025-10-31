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

# Run the test with the specific failing input
try:
    ext = Extension(
        'A',
        ['A.pyx'],
        pyrex_gdb=True,
        cython_include_dirs=['0'],
        cython_directives={'A': False}
    )

    assert ext.cython_include_dirs == ['0'], f"Expected ['0'], got {ext.cython_include_dirs}"
    assert ext.cython_directives == {'A': False}, f"Expected {{'A': False}}, got {ext.cython_directives}"
    print("Test failed: Assertions did not raise (bug not detected)")
except AssertionError as e:
    print(f"AssertionError: {e}")