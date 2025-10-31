import datetime

# Test the exact truncation logic from Django's code
dt = datetime.datetime(2000, 1, 1, 12, 34, 56, 123456, tzinfo=datetime.timezone.utc)
r = dt.isoformat()
print(f"Original isoformat: {r}")
print(f"Length: {len(r)}")
print(f"Character breakdown:")
for i, char in enumerate(r):
    print(f"  [{i:2d}]: '{char}'")

print(f"\nTruncation logic for datetime: r[:23] + r[26:]")
if dt.microsecond:
    truncated = r[:23] + r[26:]
    print(f"Result: {truncated}")
    print(f"What's lost: r[23:26] = '{r[23:26]}'")

print("\n" + "="*50)

# Test for time
t = datetime.time(12, 34, 56, 123456)
r_time = t.isoformat()
print(f"\nOriginal time isoformat: {r_time}")
print(f"Length: {len(r_time)}")
print(f"Character breakdown:")
for i, char in enumerate(r_time):
    print(f"  [{i:2d}]: '{char}'")

print(f"\nTruncation logic for time: r[:12]")
if t.microsecond:
    truncated_time = r_time[:12]
    print(f"Result: {truncated_time}")
    print(f"What's lost: r[12:] = '{r_time[12:]}'")

# Verify the exact index positions
print("\n" + "="*50)
print("\nVerifying index positions:")
dt_str = "2000-01-01T12:34:56.123456+00:00"
print(f"Full datetime string: {dt_str}")
print(f"Positions 0-23:  '{dt_str[:23]}'")
print(f"Positions 23-26: '{dt_str[23:26]}'")
print(f"Positions 26+:   '{dt_str[26:]}'")

time_str = "12:34:56.123456"
print(f"\nFull time string: {time_str}")
print(f"Positions 0-12:  '{time_str[:12]}'")
print(f"Positions 12+:   '{time_str[12:]}'")