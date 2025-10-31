from hypothesis import given, strategies as st, settings
from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


@given(
    st.one_of(st.just(0), st.just(False), st.just(""), st.just([]))
)
@settings(max_examples=100)
def test_get_extension_attr_falsy_bug(falsy_value):
    dist = Distribution()
    build_ext_instance = build_ext(dist)
    build_ext_instance.initialize_options()
    build_ext_instance.finalize_options()

    class MockExtension:
        pass

    ext = MockExtension()

    setattr(build_ext_instance, 'test_option', falsy_value)
    setattr(ext, 'test_option', "extension_value")

    result = build_ext_instance.get_extension_attr(ext, 'test_option', default="default")

    print(f"Testing falsy_value={falsy_value!r}, got result={result!r}, expected={falsy_value!r}")
    assert result == falsy_value, f"Expected {falsy_value!r}, but got {result!r}"

if __name__ == "__main__":
    test_get_extension_attr_falsy_bug()