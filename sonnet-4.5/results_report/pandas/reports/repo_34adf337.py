from pandas.io.common import is_url

# Test with a malformed URL containing unmatched bracket
print("Testing is_url('http://['):")
result = is_url("http://[")