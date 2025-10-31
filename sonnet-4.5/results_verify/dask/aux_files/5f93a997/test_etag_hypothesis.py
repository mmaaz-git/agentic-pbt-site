import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example, settings
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles


@settings(max_examples=1000)
@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=('Cs', 'Cc'),
            blacklist_characters=(',', '\r', '\n')
        ),
        min_size=1
    )
)
@example("W")
@example("/")
@example("W/")
@example("testW")
@example("test/")
def test_weak_etag_matching_property(etag_value):
    sf = StaticFiles(directory="/tmp", check_dir=False)

    strong_etag = f'"{etag_value}"'
    weak_etag = f'W/{strong_etag}'

    response_headers = Headers({"etag": strong_etag})
    request_headers = Headers({"if-none-match": weak_etag})

    result = sf.is_not_modified(response_headers, request_headers)

    assert result, f"Weak ETag {weak_etag} should match strong ETag {strong_etag}, but doesn't"

if __name__ == "__main__":
    test_weak_etag_matching_property()