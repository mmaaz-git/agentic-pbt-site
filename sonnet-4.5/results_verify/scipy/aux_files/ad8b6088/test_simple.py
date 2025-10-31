import numpy as np
import tempfile
from scipy.io import wavfile
import os

data_2d = np.array([[100]], dtype=np.int16)

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    wavfile.write(f.name, 1000, data_2d)
    rate, loaded = wavfile.read(f.name)
    temp_file = f.name

print(f"Written: {data_2d.shape}")
print(f"Read:    {loaded.shape}")

try:
    assert data_2d.shape == loaded.shape, f"Shape mismatch: wrote {data_2d.shape}, read {loaded.shape}"
    print("Test passed!")
except AssertionError as e:
    print(f"AssertionError: {e}")

os.unlink(temp_file)