#!/usr/bin/env python3
"""Test script to reproduce the redact_data bug"""

from llm.default_plugins.openai_models import redact_data

# Test case from the bug report
test_input = {
    "image_url": {
        "url": "data:image/png;base64,abc123",
        "nested": {
            "image_url": {
                "url": "data:image/png;base64,xyz789"
            }
        }
    }
}

print("Original input:")
print(test_input)

result = redact_data(test_input)

print("\nResult after redact_data:")
print(result)

print("\nAssertion tests:")
try:
    assert result["image_url"]["url"] == "data:..."
    print("✓ First assertion passed: Top-level URL was redacted")
except AssertionError:
    print("✗ First assertion failed: Top-level URL was NOT redacted")

try:
    assert result["image_url"]["nested"]["image_url"]["url"] == "data:..."
    print("✓ Second assertion passed: Nested URL was redacted")
except AssertionError:
    print("✗ Second assertion failed: Nested URL was NOT redacted")
    print(f"  Actual nested URL value: {result['image_url']['nested']['image_url']['url']}")