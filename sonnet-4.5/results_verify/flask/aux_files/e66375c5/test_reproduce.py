import pandas as pd

print("Testing NA comparisons:")
print("=" * 50)

result_none = pd.NA == None
result_list = pd.NA == []
result_dict = pd.NA == {}
result_tuple = pd.NA == ()
result_set = pd.NA == set()
result_object = pd.NA == object()
result_int = pd.NA == 0
result_str = pd.NA == ""
result_bool = pd.NA == True
result_na = pd.NA == pd.NA

print(f"NA == None: {result_none} (type: {type(result_none).__name__})")
print(f"NA == []: {result_list} (type: {type(result_list).__name__})")
print(f"NA == {{}}: {result_dict} (type: {type(result_dict).__name__})")
print(f"NA == (): {result_tuple} (type: {type(result_tuple).__name__})")
print(f"NA == set(): {result_set} (type: {type(result_set).__name__})")
print(f"NA == object(): {result_object} (type: {type(result_object).__name__})")
print(f"NA == 0: {result_int} (type: {type(result_int).__name__})")
print(f"NA == '': {result_str} (type: {type(result_str).__name__})")
print(f"NA == True: {result_bool} (type: {type(result_bool).__name__})")
print(f"NA == pd.NA: {result_na} (type: {type(result_na).__name__})")

print("\nTesting NA inequality comparisons:")
print("=" * 50)

result_ne_none = pd.NA != None
result_ne_list = pd.NA != []
result_ne_int = pd.NA != 0

print(f"NA != None: {result_ne_none} (type: {type(result_ne_none).__name__})")
print(f"NA != []: {result_ne_list} (type: {type(result_ne_list).__name__})")
print(f"NA != 0: {result_ne_int} (type: {type(result_ne_int).__name__})")