from pandas.core.computation.common import ensure_decoded

data = b'\x80'
result = ensure_decoded(data)
print(f"Result: {result}")