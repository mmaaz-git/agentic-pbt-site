from django.db.backends.utils import truncate_name

identifier = '00'
length = 1
hash_len = 2

result = truncate_name(identifier, length=length, hash_len=hash_len)

print(f"Input: identifier={identifier!r}, length={length}, hash_len={hash_len}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")
print(f"Expected: output length <= {length}")
print(f"Actual: output length = {len(result)}")
print(f"Bug: Output length ({len(result)}) exceeds requested length ({length})")