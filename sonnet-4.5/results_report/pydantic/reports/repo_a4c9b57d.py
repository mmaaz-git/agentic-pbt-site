import pickle
from pydantic.deprecated.parse import load_str_bytes, Protocol

data = []
pickled_bytes = pickle.dumps(data)
pickled_str = pickled_bytes.decode('latin1')

print(f"Original pickle bytes: {pickled_bytes}")
print(f"Decoded with latin1: {repr(pickled_str)}")

try:
    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                            encoding='latin1', allow_pickle=True)
    print(f"Successfully loaded: {result}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")