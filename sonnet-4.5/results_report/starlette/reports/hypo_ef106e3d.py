from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.sampled_from(list(SAFELISTED_HEADERS)), min_size=1, max_size=3))
def test_cors_no_duplicate_headers_when_user_provides_safelisted(safelisted_subset):
    user_headers = [h.lower() for h in safelisted_subset]

    cors = CORSMiddleware(
        dummy_app,
        allow_origins=["*"],
        allow_headers=user_headers,
        allow_methods=["GET"]
    )

    assert len(cors.allow_headers) == len(set(cors.allow_headers)), \
        f"Duplicate headers: {cors.allow_headers}"


if __name__ == "__main__":
    test_cors_no_duplicate_headers_when_user_provides_safelisted()