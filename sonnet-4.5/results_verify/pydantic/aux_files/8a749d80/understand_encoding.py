import pickle

# Create pickle data
data = []
pickled_bytes = pickle.dumps(data)
print(f"Original pickle bytes: {pickled_bytes}")

# Decode using latin1
pickled_str = pickled_bytes.decode('latin1')
print(f"String decoded with latin1: {repr(pickled_str)}")

# When re-encoded with UTF-8 (default), it gets mangled
utf8_encoded = pickled_str.encode()  # default is UTF-8
print(f"Re-encoded with UTF-8: {utf8_encoded}")

# When re-encoded with latin1, it's correct
latin1_encoded = pickled_str.encode('latin1')
print(f"Re-encoded with latin1: {latin1_encoded}")

# Verify that the latin1 encoding gives us back the original bytes
print(f"Latin1 encoding matches original: {latin1_encoded == pickled_bytes}")

# Try to unpickle each
print("\nUnpickling UTF-8 encoded:")
try:
    pickle.loads(utf8_encoded)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")

print("\nUnpickling latin1 encoded:")
try:
    result = pickle.loads(latin1_encoded)
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {e}")