import scipy.io.matlab as mat
import numpy as np
import tempfile
import os

text = 'hello'
test_dict = {'text': text}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp1 = f.name
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp2 = f.name
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp3 = f.name

try:
    mat.savemat(temp1, test_dict)
    loaded1 = mat.loadmat(temp1, chars_as_strings=True)
    print(f"1st load type: {type(loaded1['text'])}")
    print(f"1st load value: {loaded1['text']}")

    user_data1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
    mat.savemat(temp2, user_data1)
    loaded2 = mat.loadmat(temp2, chars_as_strings=True)
    print(f"\n2nd load type: {type(loaded2['text'])}")
    print(f"2nd load value: {loaded2['text']}")

    user_data2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}
    mat.savemat(temp3, user_data2)
    loaded3 = mat.loadmat(temp3, chars_as_strings=True)
    print(f"\n3rd load type: {type(loaded3['text'])}")
    print(f"3rd load value: {loaded3['text']}")

    # Check equality
    print(f"\nloaded1['text'] == loaded2['text']: {loaded1['text'] == loaded2['text']}")
    print(f"loaded2['text'] == loaded3['text']: {loaded2['text'] == loaded3['text']}")

finally:
    os.remove(temp1)
    os.remove(temp2)
    os.remove(temp3)