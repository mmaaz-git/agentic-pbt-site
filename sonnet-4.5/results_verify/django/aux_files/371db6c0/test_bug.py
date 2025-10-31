#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

# First, let's reproduce the specific bug case
def test_specific_case():
    print("Testing specific case with CONTENT_LENGTH=-1:")
    stdin = BytesIO(b"request body data")
    stdout = BytesIO()
    stderr = BytesIO()
    environ = {"CONTENT_LENGTH": "-1", "REQUEST_METHOD": "POST"}

    handler = ServerHandler(stdin, stdout, stderr, environ)

    print(f"LimitedStream limit: {handler.stdin.limit}")
    assert handler.stdin.limit == -1, f"Expected -1 but got {handler.stdin.limit}"
    print("✓ Bug confirmed: Negative content length is not normalized to 0")
    return True

# Now run the hypothesis test
@given(st.integers())
def test_serverhandler_content_length_parsing_integers(content_length):
    stdin = BytesIO(b"test data")
    stdout = BytesIO()
    stderr = BytesIO()

    environ = {"CONTENT_LENGTH": str(content_length), "REQUEST_METHOD": "POST"}

    handler = ServerHandler(stdin, stdout, stderr, environ)

    expected_limit = max(0, content_length)
    assert handler.stdin.limit == expected_limit, f"For content_length={content_length}, expected {expected_limit} but got {handler.stdin.limit}"

if __name__ == "__main__":
    # Test the specific case
    try:
        test_specific_case()
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    # Run hypothesis test
    print("\nRunning hypothesis test...")
    try:
        test_serverhandler_content_length_parsing_integers()
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")