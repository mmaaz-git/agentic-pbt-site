from Cython.Plex.Regexps import RawCodeRange

rcr = RawCodeRange(50, 60)

# Check what the correct string representation would be with the fix
print(f"rcr.range = {rcr.range}")
print(f"Correct string would be: CodeRange(%d,%d)" % rcr.range)