import pickle
import warnings
from hypothesis import given, settings, strategies as st
from pydantic.deprecated.parse import load_str_bytes, Protocol

warnings.filterwarnings('ignore', category=DeprecationWarning)

@settings(max_examples=500)
@given(st.lists(st.integers()))
def test_load_str_bytes_pickle_encoding_parameter(lst):
    pickled_bytes = pickle.dumps(lst)
    pickled_str = pickled_bytes.decode('latin1')

    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                          encoding='latin1', allow_pickle=True)
    assert result == lst

if __name__ == "__main__":
    test_load_str_bytes_pickle_encoding_parameter()