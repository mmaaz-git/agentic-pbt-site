import numpy as np
import tempfile
from scipy.io import wavfile
import os

print("Testing different channel counts:")
for n_channels in [1, 2, 3, 4]:
    data = np.random.randint(-1000, 1000, (100, n_channels), dtype=np.int16)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wavfile.write(f.name, 8000, data)
        rate, loaded = wavfile.read(f.name)
        temp_file = f.name

    print(f"  {n_channels} channel(s): Written shape {data.shape} → Read shape {loaded.shape}")

    if data.shape == loaded.shape:
        print(f"    ✓ Shapes match")
    else:
        print(f"    ✗ Shape mismatch!")

    os.unlink(temp_file)

print("\nTesting 1D mono input:")
data_1d = np.array([100, 200, 300], dtype=np.int16)

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    wavfile.write(f.name, 8000, data_1d)
    rate, loaded_1d = wavfile.read(f.name)
    temp_file = f.name

print(f"  1D input: Written shape {data_1d.shape} → Read shape {loaded_1d.shape}")

if data_1d.shape == loaded_1d.shape:
    print(f"    ✓ Shapes match")
else:
    print(f"    ✗ Shape mismatch!")

os.unlink(temp_file)