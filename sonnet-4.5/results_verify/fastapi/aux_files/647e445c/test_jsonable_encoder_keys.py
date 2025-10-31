from fastapi.encoders import jsonable_encoder
import json

# Test how jsonable_encoder handles different key types

# Test 1: String keys (should work)
test1 = {"key": "value"}
result1 = jsonable_encoder(test1)
print(f"Test 1 - String keys: {result1}")
print(f"  JSON encodable: {json.dumps(result1)}")

# Test 2: Integer keys
test2 = {1: "value", 2: "another"}
result2 = jsonable_encoder(test2)
print(f"\nTest 2 - Integer keys: {result2}")
print(f"  Key types: {[type(k).__name__ for k in result2.keys()]}")
print(f"  JSON encodable: {json.dumps(result2)}")

# Test 3: Float keys
test3 = {3.14: "value"}
result3 = jsonable_encoder(test3)
print(f"\nTest 3 - Float keys: {result3}")
print(f"  Key types: {[type(k).__name__ for k in result3.keys()]}")
print(f"  JSON encodable: {json.dumps(result3)}")

# Test 4: Boolean keys
test4 = {True: "value1", False: "value2"}
result4 = jsonable_encoder(test4)
print(f"\nTest 4 - Boolean keys: {result4}")
print(f"  Key types: {[type(k).__name__ for k in result4.keys()]}")
print(f"  JSON encodable: {json.dumps(result4)}")

# Test 5: None key
test5 = {None: "value"}
result5 = jsonable_encoder(test5)
print(f"\nTest 5 - None key: {result5}")
print(f"  Key types: {[type(k).__name__ for k in result5.keys()]}")
print(f"  JSON encodable: {json.dumps(result5)}")

# Test 6: Tuple keys (the bug case)
test6 = {(1, 2): "value"}
try:
    result6 = jsonable_encoder(test6)
    print(f"\nTest 6 - Tuple keys: {result6}")
    print(f"  Key types: {[type(k).__name__ for k in result6.keys()]}")
    print(f"  JSON encodable: {json.dumps(result6)}")
except TypeError as e:
    print(f"\nTest 6 - Tuple keys failed: {e}")

# Test 7: Empty tuple key
test7 = {(): "value"}
try:
    result7 = jsonable_encoder(test7)
    print(f"\nTest 7 - Empty tuple key: {result7}")
    print(f"  Key types: {[type(k).__name__ for k in result7.keys()]}")
    print(f"  JSON encodable: {json.dumps(result7)}")
except TypeError as e:
    print(f"\nTest 7 - Empty tuple key failed: {e}")

# Test 8: Single element tuple key
test8 = {(1,): "value"}
try:
    result8 = jsonable_encoder(test8)
    print(f"\nTest 8 - Single tuple key: {result8}")
    print(f"  Key types: {[type(k).__name__ for k in result8.keys()]}")
    print(f"  JSON encodable: {json.dumps(result8)}")
except TypeError as e:
    print(f"\nTest 8 - Single tuple key failed: {e}")