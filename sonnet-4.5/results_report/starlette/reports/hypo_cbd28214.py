from hypothesis import given, strategies as st
from starlette.middleware import Middleware


@given(
    st.text(),
    st.lists(st.one_of(st.integers(), st.text(), st.booleans()), min_size=1),
    st.dictionaries(
        st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.one_of(st.integers(), st.text(), st.booleans()),
        min_size=0
    )
)
def test_middleware_repr_no_leading_comma(name, args, kwargs):
    class TestMiddleware:
        pass

    TestMiddleware.__name__ = name

    m = Middleware(TestMiddleware, *args, **kwargs)
    repr_str = repr(m)

    assert not repr_str.startswith("Middleware(, "), (
        f"Repr should not have leading comma, got: {repr_str}"
    )

if __name__ == "__main__":
    test_middleware_repr_no_leading_comma()