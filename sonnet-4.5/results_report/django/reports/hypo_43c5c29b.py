from hypothesis import given, strategies as st
from django.utils.http import quote_etag


@given(st.text())
def test_quote_etag_idempotence(s):
    result1 = quote_etag(s)
    result2 = quote_etag(result1)
    assert result1 == result2, f"quote_etag is not idempotent: quote_etag({s!r}) = {result1!r}, but quote_etag({result1!r}) = {result2!r}"

# Run the test
if __name__ == "__main__":
    test_quote_etag_idempotence()