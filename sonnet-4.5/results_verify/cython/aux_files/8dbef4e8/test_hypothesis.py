#!/usr/bin/env python3
"""Hypothesis test for the redact_data bug"""

from hypothesis import given, strategies as st, settings
from llm.default_plugins.openai_models import redact_data


def find_data_urls(obj, path=""):
    urls = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if key == "image_url" and isinstance(value, dict) and "url" in value:
                if isinstance(value["url"], str) and value["url"].startswith("data:"):
                    urls.append((current_path + ".url", value["url"]))
            urls.extend(find_data_urls(value, current_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            urls.extend(find_data_urls(item, f"{path}[{i}]"))
    return urls


@given(st.integers(min_value=2, max_value=3))
@settings(max_examples=100)
def test_redact_data_nested_image_urls(nesting_level):
    data = {}
    current = data
    for i in range(nesting_level):
        current["image_url"] = {
            "url": f"data:image/png;base64,nested{i}",
            "other": f"value{i}"
        }
        if i < nesting_level - 1:
            current["image_url"]["child"] = {}
            current = current["image_url"]["child"]

    print(f"\nTesting with nesting_level={nesting_level}")
    print(f"Input data: {data}")

    result = redact_data(data)
    print(f"Result: {result}")

    urls_after = find_data_urls(result)
    print(f"URLs found after redaction: {urls_after}")

    non_redacted = [(path, url) for path, url in urls_after if url != "data:..."]

    if non_redacted:
        print(f"ERROR: Found {len(non_redacted)} non-redacted URLs:")
        for path, url in non_redacted:
            print(f"  - {path}: {url}")

    assert len(non_redacted) == 0, f"Found {len(non_redacted)} non-redacted URLs"


# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    test_redact_data_nested_image_urls()
    print("\nAll tests passed!")