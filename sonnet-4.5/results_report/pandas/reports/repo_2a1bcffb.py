from pandas.core.dtypes.common import ensure_str

invalid_utf8_bytes = b'\x80'
result = ensure_str(invalid_utf8_bytes)
print(f"Result: {result}")