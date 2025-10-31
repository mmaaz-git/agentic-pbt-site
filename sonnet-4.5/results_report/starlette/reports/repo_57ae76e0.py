from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, "test.html"), 'w') as f:
        f.write("<html><body>{{ request.url }}</body></html>")

    templates = Jinja2Templates(directory=tmpdir)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("localhost", 8000),
    }
    request = Request(scope)

    response = templates.TemplateResponse(
        request=request,
        name="test.html",
        context=None
    )

    print("Response created successfully")
    print(f"Status code: {response.status_code}")