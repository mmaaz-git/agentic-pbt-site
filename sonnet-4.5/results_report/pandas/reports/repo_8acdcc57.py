import pandas.io.common as common

url = "http://["
result = common.is_url(url)
print(f"Result: {result}")