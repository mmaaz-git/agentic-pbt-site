import io
from Cython.Plex import *

# Create a simple lexicon with Eof pattern and action
lexicon = Lexicon([
    (Str('hello'), 'HELLO_TOKEN'),
    (Eof, 'EOF_TOKEN')
])

# Let's inspect the lexicon's machine to see if Eof action is registered
machine = lexicon.machine
print("Lexicon machine states:")
for state_name, state in machine.initial_states.items():
    print(f"  State '{state_name}': {state}")

# Create a scanner with debug enabled
scanner = Scanner(lexicon, io.StringIO('hello'))
scanner.trace = 1  # Enable tracing

print("\nScanning 'hello':")
token1, text1 = scanner.read()
print(f"Got token: {token1!r}, text: {text1!r}")

print("\nScanning EOF:")
token2, text2 = scanner.read()
print(f"Got EOF token: {token2!r}, text: {text2!r}")
print(f"Expected: 'EOF_TOKEN', Actual: {token2!r}")