from Cython.Plex.Regexps import RawCodeRange

# Create a RawCodeRange instance with valid code range
rcr = RawCodeRange(50, 60)

# Try to get string representation
try:
    s = str(rcr)
    print(f"String representation: {s}")
except AttributeError as e:
    print(f"AttributeError: {e}")
    print(f"rcr.range exists: {hasattr(rcr, 'range')}")
    print(f"rcr.range value: {rcr.range if hasattr(rcr, 'range') else 'N/A'}")
    print(f"rcr.code1 exists: {hasattr(rcr, 'code1')}")
    print(f"rcr.code2 exists: {hasattr(rcr, 'code2')}")