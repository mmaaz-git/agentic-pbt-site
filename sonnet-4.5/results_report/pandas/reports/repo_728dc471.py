from io import StringIO
from Cython.Plex import Lexicon, Range, Rep, Scanner

# Create a lexer that accepts zero or more digits
digit = Range('0', '9')
lexicon = Lexicon([(Rep(digit), 'INT')])

# Give it input that contains non-digit characters
scanner = Scanner(lexicon, StringIO('abc'), 'test')

# Try to read tokens - this will loop infinitely
print("Attempting to read tokens from 'abc' with Rep(digit) pattern:")
for i in range(10):
    token_type, token_text = scanner.read()
    print(f'Iteration {i}: token_type={token_type!r}, token_text={token_text!r}')
    if token_type is None:
        print("Reached end of file")
        break
else:
    print("Stopped after 10 iterations to prevent infinite loop")