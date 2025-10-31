from hypothesis import given, strategies as st
from Cython.Distutils import Extension


@given(
    st.lists(st.text(min_size=1), min_size=1),
    st.dictionaries(st.text(min_size=1), st.integers()),
)
def test_extension_cython_params_preserved_with_pyrex_kwargs(sources, directives):
    ext = Extension(
        name="test",
        sources=sources,
        cython_directives=directives,
        pyrex_cplus=True,
    )

    assert ext.cython_directives == directives


if __name__ == "__main__":
    test_extension_cython_params_preserved_with_pyrex_kwargs()