import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

# The bug report claims 100 * 2**50 produces '100.00 PiB' with length 11
# But my test shows length 10. Let me verify character by character.

value = 100 * 2**50
result = format_bytes(value)
print(f"Testing value: {value}")
print(f"Result: '{result}'")
print(f"Result repr: {repr(result)}")
print(f"Length: {len(result)}")
print("Character breakdown:")
for i, char in enumerate(result):
    print(f"  [{i}]: '{char}' (ord={ord(char)})")

print("\n" + "="*50)

# Now test 1000 * 2**50 which should definitely be > 10 chars
value2 = 1000 * 2**50
if value2 < 2**60:
    result2 = format_bytes(value2)
    print(f"\nTesting value: {value2}")
    print(f"Result: '{result2}'")
    print(f"Length: {len(result2)}")
    print("Character breakdown:")
    for i, char in enumerate(result2):
        print(f"  [{i}]: '{char}' (ord={ord(char)})")

print("\n" + "="*50)

# And test the max value < 2**60
value3 = 2**60 - 1
result3 = format_bytes(value3)
print(f"\nTesting value: {value3}")
print(f"Result: '{result3}'")
print(f"Length: {len(result3)}")
print("Character breakdown:")
for i, char in enumerate(result3):
    print(f"  [{i}]: '{char}' (ord={ord(char)})")

# Let's check when we get to 11 characters
print("\n" + "="*50)
print("\nLooking for first 11-character output:")
for mult in range(100, 1100, 100):
    value = mult * 2**50
    if value < 2**60:
        result = format_bytes(value)
        print(f"{mult} * 2**50: '{result}' (length: {len(result)})")
        if len(result) == 11:
            print(f"  --> First 11-char at {mult} * 2**50")
            break