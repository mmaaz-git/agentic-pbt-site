"""Test to reproduce the bug reported for Jinja2Templates.TemplateResponse"""

from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os
import traceback

def test_simple_reproduction():
    """Simple test case to reproduce the bug"""
    print("Testing simple reproduction case...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test template
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

        try:
            # Test with context=None explicitly passed
            response = templates.TemplateResponse(
                request=request,
                name="test.html",
                context=None
            )
            print("SUCCESS: Response created with context=None")
            print(f"Response status code: {response.status_code}")
        except AttributeError as e:
            print(f"ERROR: AttributeError occurred as reported: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            traceback.print_exc()
            return False

    return True

def test_context_not_provided():
    """Test when context is not provided at all (using default)"""
    print("\nTesting when context is not provided...")

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

        try:
            # Test without providing context at all (should use default)
            response = templates.TemplateResponse(
                request=request,
                name="test.html"
                # context not provided - should use default
            )
            print("SUCCESS: Response created without context parameter")
            print(f"Response status code: {response.status_code}")
        except Exception as e:
            print(f"ERROR: Exception occurred: {e}")
            traceback.print_exc()
            return False

    return True

def test_context_empty_dict():
    """Test when context is explicitly an empty dict"""
    print("\nTesting with explicit empty dict...")

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

        try:
            # Test with explicit empty dict
            response = templates.TemplateResponse(
                request=request,
                name="test.html",
                context={}
            )
            print("SUCCESS: Response created with context={}")
            print(f"Response status code: {response.status_code}")
        except Exception as e:
            print(f"ERROR: Exception occurred: {e}")
            traceback.print_exc()
            return False

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Jinja2Templates.TemplateResponse bug report")
    print("=" * 60)

    # Run tests
    results = []
    results.append(("Simple reproduction (context=None)", test_simple_reproduction()))
    results.append(("Context not provided", test_context_not_provided()))
    results.append(("Empty dict context", test_context_empty_dict()))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")