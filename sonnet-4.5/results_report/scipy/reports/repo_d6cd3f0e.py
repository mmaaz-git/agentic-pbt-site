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
    # First save and load
    mat.savemat(temp1, test_dict)
    loaded1 = mat.loadmat(temp1, chars_as_strings=False)
    print(f"1st load shape: {loaded1['text'].shape}")
    print(f"1st load type: {type(loaded1['text'])}")
    print(f"1st load content: {loaded1['text']}")
    print()

    # Second save and load
    user_data1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
    mat.savemat(temp2, user_data1)
    loaded2 = mat.loadmat(temp2, chars_as_strings=False)
    print(f"2nd load shape: {loaded2['text'].shape}")
    print(f"2nd load type: {type(loaded2['text'])}")
    print(f"2nd load content shape details: {loaded2['text']}")
    print()

    # Third save and load
    user_data2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}
    mat.savemat(temp3, user_data2)
    loaded3 = mat.loadmat(temp3, chars_as_strings=False)
    print(f"3rd load shape: {loaded3['text'].shape}")
    print(f"3rd load type: {type(loaded3['text'])}")
    print(f"3rd load content shape details: {loaded3['text']}")

    # Verify arrays are not equal after round-trip
    print(f"\nArrays equal after 1st and 2nd load? {np.array_equal(loaded1['text'], loaded2['text'])}")

finally:
    os.remove(temp1)
    os.remove(temp2)
    os.remove(temp3)