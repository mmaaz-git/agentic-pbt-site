from hypothesis import given, strategies as st
from fastapi.openapi.utils import get_openapi

@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    terms_of_service=st.one_of(st.none(), st.text(max_size=200))
)
def test_get_openapi_terms_preservation(title, version, terms_of_service):
    result = get_openapi(title=title, version=version, terms_of_service=terms_of_service, routes=[])

    if terms_of_service is not None:
        assert "termsOfService" in result["info"], f"termsOfService missing when terms_of_service={repr(terms_of_service)}"
        assert result["info"]["termsOfService"] == terms_of_service

# Run the test
if __name__ == "__main__":
    test_get_openapi_terms_preservation()