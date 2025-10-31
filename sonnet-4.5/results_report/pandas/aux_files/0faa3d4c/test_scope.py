from pandas.api.types import pandas_dtype
import numpy as np

test_cases = [
    {'0': ''},
    {'a': 1},
    {'b': 'text'},
    {'foo': 'bar'},
    {'key': [1, 2]},
    {'x': (1,)},
    {'y': (1, 2, 3, 4)},  # Invalid tuple size
]

for test_input in test_cases:
    try:
        result = pandas_dtype(test_input)
        print(f"Input {test_input}: Successfully converted to {result}")
    except ValueError as e:
        print(f"Input {test_input}: ValueError - {e}")
    except TypeError as e:
        print(f"Input {test_input}: TypeError - {e}")
    except Exception as e:
        print(f"Input {test_input}: {type(e).__name__} - {e}")