import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from hypothesis import given, strategies as st, assume
from fastapi.openapi.docs import get_swagger_ui_html

@given(st.text())
def test_oauth2_redirect_url_quote_injection(url):
    assume("'" in url or '"' in url or "\n" in url)

    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Test",
        oauth2_redirect_url=url
    )

    html_str = html.body.decode()
    oauth_line_start = html_str.find("oauth2RedirectUrl")
    if oauth_line_start == -1:
        return

    oauth_line_end = html_str.find("\n", oauth_line_start)
    oauth_line = html_str[oauth_line_start:oauth_line_end]

    if "'" in url:
        assert "\\'" in oauth_line or "\\x" in oauth_line, \
            f"Single quote in URL not properly escaped: {repr(url)} -> {repr(oauth_line)}"

if __name__ == "__main__":
    test_oauth2_redirect_url_quote_injection()