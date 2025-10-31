import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

# Create middleware with lowercase "accept" header
middleware = CORSMiddleware(dummy_app, allow_headers=["accept"])

# Print the resulting allow_headers list
print("allow_headers:", middleware.allow_headers)

# Check for duplicates
unique_headers = set(middleware.allow_headers)
if len(middleware.allow_headers) != len(unique_headers):
    print(f"ERROR: Duplicate headers found!")
    print(f"  List length: {len(middleware.allow_headers)}")
    print(f"  Unique count: {len(unique_headers)}")

    # Find and print duplicates
    from collections import Counter
    header_counts = Counter(middleware.allow_headers)
    duplicates = {h: count for h, count in header_counts.items() if count > 1}
    print(f"  Duplicates: {duplicates}")
else:
    print("No duplicates found")