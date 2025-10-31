"""Test to reproduce the API key whitespace validation bug"""
from unittest.mock import Mock
import pytest
from fastapi.security.api_key import APIKeyHeader
from hypothesis import given, settings, strategies as st
from starlette.requests import Request
from starlette.exceptions import HTTPException


# Property-based test from bug report
@pytest.mark.asyncio
@given(st.sampled_from([" ", "  ", "\t", "\n", "\r", "   ", " \t ", "\t\n", " \n\r\t "]))
@settings(max_examples=20, deadline=None)
async def test_api_key_whitespace_only_should_be_rejected(whitespace_key):
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": whitespace_key}

    result = await api_key(request)

    assert result is None or result.strip() != "", \
        f"Whitespace-only API key {whitespace_key!r} should be rejected, but got {result!r}"


# Direct reproduction tests
@pytest.mark.asyncio
async def test_empty_string_is_rejected():
    """Test that empty string is correctly rejected with 403 error"""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": ""}

    with pytest.raises(HTTPException) as exc:
        await api_key(request)

    assert exc.value.status_code == 403
    print("✓ Empty string is rejected with 403 error")


@pytest.mark.asyncio
async def test_whitespace_is_accepted():
    """Test that whitespace-only strings are incorrectly accepted"""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": " "}

    result = await api_key(request)
    assert result == " "
    print(f"✗ Whitespace string ' ' is accepted and returned as: {result!r}")


@pytest.mark.asyncio
async def test_various_whitespace():
    """Test various whitespace strings"""
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

    whitespace_cases = [
        " ",      # single space
        "  ",     # multiple spaces
        "\t",     # tab
        "\n",     # newline
        "\r",     # carriage return
        " \t\n ", # mixed whitespace
    ]

    for ws in whitespace_cases:
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": ws}
        result = await api_key(request)
        print(f"Whitespace {ws!r} -> Result: {result!r} (Expected: None)")


@pytest.mark.asyncio
async def test_none_is_rejected():
    """Test that None/missing header is rejected"""
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
    request = Mock(spec=Request)
    request.headers = {}  # No X-API-Key header

    result = await api_key(request)
    assert result is None
    print("✓ Missing header returns None")


@pytest.mark.asyncio
async def test_valid_api_key():
    """Test that valid non-empty API keys work correctly"""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": "valid-key-123"}

    result = await api_key(request)
    assert result == "valid-key-123"
    print("✓ Valid API key is accepted correctly")


if __name__ == "__main__":
    import asyncio

    async def main():
        print("\n=== Testing API Key Whitespace Validation ===\n")

        # Run individual tests
        try:
            await test_empty_string_is_rejected()
        except Exception as e:
            print(f"Empty string test: {e}")

        try:
            await test_whitespace_is_accepted()
        except Exception as e:
            print(f"Whitespace test: {e}")

        await test_various_whitespace()
        await test_none_is_rejected()
        await test_valid_api_key()

        print("\n=== Property-based test with hypothesis ===\n")
        # Run one example from property test
        api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": " "}
        result = await api_key(request)
        print(f"Property test example: whitespace ' ' -> {result!r}")
        if result is not None and result.strip() == "":
            print("BUG CONFIRMED: Whitespace-only API key was accepted!")

    asyncio.run(main())