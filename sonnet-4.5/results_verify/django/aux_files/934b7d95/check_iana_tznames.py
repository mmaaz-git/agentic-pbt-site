import zoneinfo
import re

# Get all available timezone names
all_timezones = sorted(zoneinfo.available_timezones())

# Check what characters are actually used in timezone names
used_chars = set()
for tz in all_timezones:
    used_chars.update(tz)

print("Total number of timezones:", len(all_timezones))
print("\nSample timezone names:")
for tz in all_timezones[:10]:
    print(f"  {tz}")

print("\nAll unique characters used in timezone names:")
print(''.join(sorted(used_chars)))

# Check if any timezone contains non-ASCII characters
non_ascii_timezones = []
for tz in all_timezones:
    if any(ord(c) > 127 for c in tz):
        non_ascii_timezones.append(tz)

if non_ascii_timezones:
    print("\nTimezones with non-ASCII characters:")
    for tz in non_ascii_timezones:
        print(f"  {tz}")
else:
    print("\nNo timezones contain non-ASCII characters")

# Test the regex pattern from Django
django_regex = re.compile(r"^[\w/:+-]+$")

print("\n\nTesting Django regex against all timezone names:")
failed_matches = []
for tz in all_timezones:
    if not django_regex.match(tz):
        failed_matches.append(tz)

if failed_matches:
    print(f"Django regex fails on {len(failed_matches)} timezone names:")
    for tz in failed_matches[:10]:
        print(f"  {tz}")
else:
    print("Django regex matches all valid timezone names")

# Check what ASCII-only regex would match
ascii_regex = re.compile(r"^[a-zA-Z0-9_/:+-]+$")
print("\n\nTesting ASCII-only regex against all timezone names:")
failed_ascii = []
for tz in all_timezones:
    if not ascii_regex.match(tz):
        failed_ascii.append(tz)

if failed_ascii:
    print(f"ASCII-only regex fails on {len(failed_ascii)} timezone names:")
    for tz in failed_ascii[:10]:
        print(f"  {tz}")
else:
    print("ASCII-only regex matches all valid timezone names")