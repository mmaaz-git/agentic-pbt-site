#!/usr/bin/env python3
"""
Reproduce the bug in Django's WSGIRequestHandler.get_environ method
which modifies a dictionary while iterating over it.

The code at django/core/servers/basehttp.py lines 220-222:
    for k in self.headers:
        if "_" in k:
            del self.headers[k]

This violates Python's rule against modifying dictionaries during iteration.
"""

# Test with regular Python dictionary (most clear demonstration)
headers_dict = {
    'X-Custom-Header': 'value1',
    'X_Forwarded_For': '192.168.1.1',  # Has underscore
    'User_Agent': 'Mozilla/5.0',       # Has underscore
    'X_Real_IP': '10.0.0.1',          # Has underscore
    'Content-Type': 'text/html'
}

print("Headers before:", list(headers_dict.keys()))
print("Headers with underscores:", [k for k in headers_dict.keys() if '_' in k])
print()

print("Executing Django's buggy pattern:")
print("for k in headers_dict:")
print("    if '_' in k:")
print("        del headers_dict[k]")
print()

try:
    for k in headers_dict:
        if "_" in k:
            del headers_dict[k]
    print("ERROR: No RuntimeError raised! But check if all underscore headers were removed...")
except RuntimeError as e:
    print(f"RuntimeError: {e}")

print()
print("Headers after:", list(headers_dict.keys()))
remaining_underscore = [k for k in headers_dict.keys() if '_' in k]
print("Headers with underscores remaining:", remaining_underscore)

if remaining_underscore:
    print()
    print("BUG CONFIRMED: Not all headers with underscores were removed!")
    print("This defeats the security measure meant to prevent header spoofing.")