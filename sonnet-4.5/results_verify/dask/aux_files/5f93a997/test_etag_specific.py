import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

# Test the specific failing case mentioned in the bug report
def test_specific_cases():
    sf = StaticFiles(directory="/tmp", check_dir=False)

    test_cases = [
        ("testW", False),  # Ends with W - should fail according to bug report
        ("test/", False),  # Ends with / - should fail according to bug report
        ("test", True),    # Normal case - should work
        ("W", False),      # Just W - should fail
        ("/", False),      # Just / - should fail
    ]

    for etag_value, should_work in test_cases:
        strong_etag = f'"{etag_value}"'
        weak_etag = f'W/{strong_etag}'

        response_headers = Headers({"etag": strong_etag})
        request_headers = Headers({"if-none-match": weak_etag})

        result = sf.is_not_modified(response_headers, request_headers)

        print(f"ETag value: {etag_value}")
        print(f"  Strong ETag: {strong_etag}")
        print(f"  Weak ETag: {weak_etag}")
        print(f"  Matches: {result}")
        print(f"  Expected to match: True")
        print(f"  Actually matches: {result}")
        if not result:
            # Show what happens with strip
            tags = [tag.strip(" W/") for tag in weak_etag.split(",")]
            print(f"  After strip(' W/'): {tags}")
        print()

if __name__ == "__main__":
    test_specific_cases()