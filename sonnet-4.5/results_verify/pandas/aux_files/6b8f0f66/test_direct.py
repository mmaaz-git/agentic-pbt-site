import pandas.io.common as common

url = "http://["
try:
    result = common.is_url(url)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Other exception: {e}")