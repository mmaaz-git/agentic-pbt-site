import pandas.io.json._json as pj

# Check if ujson is being used
print("Checking JSON backend...")
try:
    import ujson
    print(f"ujson is available: {ujson}")

    # Test ujson directly
    import json
    test_val = -9223372036854775809

    print(f"\nTesting Python's json module with {test_val}:")
    json_str = json.dumps(test_val)
    print(f"json.dumps: {json_str}")
    result = json.loads(json_str)
    print(f"json.loads: {result}")

    print(f"\nTesting ujson with {test_val}:")
    try:
        ujson_str = ujson.dumps(test_val)
        print(f"ujson.dumps: {ujson_str}")
        ujson_result = ujson.loads(ujson_str)
        print(f"ujson.loads: {ujson_result}")
    except Exception as e:
        print(f"ujson error: {e}")

except ImportError:
    print("ujson not available")