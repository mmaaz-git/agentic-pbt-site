from pandas.io.common import is_url

# Test with a malformed URL containing unmatched bracket
print("Testing is_url('http://['):")
try:
    result = is_url("http://[")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")