import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

sf = StaticFiles(directory="/tmp", check_dir=False)

strong_etag = '"testW"'
weak_etag = 'W/"testW"'

response_headers = Headers({"etag": strong_etag})
request_headers = Headers({"if-none-match": weak_etag})

result = sf.is_not_modified(response_headers, request_headers)

print(f"Strong ETag: {strong_etag}")
print(f"Weak ETag: {weak_etag}")
print(f"Matches: {result}")
print(f"Expected: True")
print(f"Actual: {result}")

if_none_match = weak_etag
tags = [tag.strip(" W/") for tag in if_none_match.split(",")]
print(f"\nAfter strip(' W/'): {tags}")
print(f"Expected: ['\"testW\"']")
print(f"Problem: strip() removes W from BOTH ends!")