from Cython.Plex.Regexps import RawCodeRange

rcr = RawCodeRange(50, 60)

try:
    s = str(rcr)
except AttributeError as e:
    print(f"AttributeError: {e}")
    print(f"rcr.range exists: {hasattr(rcr, 'range')}")
    print(f"rcr.code1 exists: {hasattr(rcr, 'code1')}")
    print(f"rcr.code2 exists: {hasattr(rcr, 'code2')}")