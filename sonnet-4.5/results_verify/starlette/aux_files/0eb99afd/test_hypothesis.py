"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os


@given(
    template_name=st.text(
        alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=50)
def test_template_response_none_context_should_not_crash(template_name):
    """
    Property: TemplateResponse should handle context=None gracefully
    since the type signature allows context: dict[str, Any] | None = None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, f"{template_name}.html")
        with open(template_path, 'w') as f:
            f.write("<html><body>{{ request.url }}</body></html>")

        templates = Jinja2Templates(directory=tmpdir)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
            "server": ("localhost", 8000),
        }
        request = Request(scope)

        response = templates.TemplateResponse(
            request=request,
            name=f"{template_name}.html",
            context=None
        )

        assert response is not None
        assert response.status_code == 200

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_template_response_none_context_should_not_crash()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()