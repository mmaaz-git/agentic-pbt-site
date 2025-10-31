import io
from Cython.Plex import *

# Simple test with ONLY an Eof pattern and nothing else on empty input
lexicon = Lexicon([
    (Str('x'), 'X'),  # Need at least one other pattern
    (Eof, 'EOF_TOKEN')
])

scanner = Scanner(lexicon, io.StringIO(''))
scanner.trace = 1  # Enable tracing

print("Scanning empty string (should match Eof):")
try:
    token, text = scanner.read()
    print(f"Got token: {token!r}, text: {text!r}")
except Exception as e:
    print(f"Exception: {e}")

print("\n" + "="*50)
print("\nNow test after reading content:")
scanner2 = Scanner(lexicon, io.StringIO('x'))
scanner2.trace = 1

print("Read 'x':")
t1, txt1 = scanner2.read()
print(f"Got: {t1!r}")

print("\nRead EOF:")
t2, txt2 = scanner2.read()
print(f"Got: {t2!r}, expected: 'EOF_TOKEN'")