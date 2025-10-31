"""Detailed trace of the issue with context=None"""

from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os

def trace_issue():
    """Trace exactly where the issue occurs"""
    print("Creating test environment...")

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.html"), 'w') as f:
            f.write("<html><body>Test</body></html>")

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

        # Let's trace what happens when we pass context=None
        print("\nCalling TemplateResponse with context=None...")
        print("This calls the method with kwargs={'request': request, 'name': 'test.html', 'context': None}")

        # The code path in TemplateResponse (line 159):
        # Since we're passing all kwargs, we go to line 188 (else clause)
        # Line 197: context = kwargs.get("context", {})
        # Since "context" key exists in kwargs with value None, get returns None (not {})

        # Simulating line 197:
        kwargs = {'request': request, 'name': 'test.html', 'context': None}
        context = kwargs.get("context", {})
        print(f"\nLine 197: kwargs.get('context', {{}}) = {context}")
        print(f"Type of context: {type(context)}")

        # Line 198 tries: request = kwargs.get("request", context.get("request"))
        # This will fail because context is None
        print("\nLine 198 tries to do: kwargs.get('request', context.get('request'))")
        print("But context is None, so context.get('request') raises AttributeError")

        try:
            request_from_context = context.get("request")
        except AttributeError as e:
            print(f"AttributeError: {e}")

        # If we got past that, line 205 would fail:
        print("\nEven if we got past line 198, line 205 would fail:")
        print("Line 205: context.setdefault('request', request)")
        print("This would also fail with AttributeError on None")

if __name__ == "__main__":
    trace_issue()