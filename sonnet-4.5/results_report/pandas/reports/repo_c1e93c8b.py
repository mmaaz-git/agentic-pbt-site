from Cython.Plex.Regexps import Range

# Test with odd-length string that should cause IndexError
s = 'abc'
result = Range(s)
print(f"Range('{s}') succeeded: {result}")