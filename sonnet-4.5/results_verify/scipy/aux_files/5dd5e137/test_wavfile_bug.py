import tempfile
import numpy as np
from scipy.io import wavfile
import os

print("=" * 60)
print("TEST 1: Simple Reproduction Case")
print("=" * 60)

rate = 44100
data_2d_single_channel = np.array([[100], [200], [300]], dtype=np.int16)

print(f"Original shape: {data_2d_single_channel.shape}")
print(f"Original data:\n{data_2d_single_channel}")

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    filename = f.name

wavfile.write(filename, rate, data_2d_single_channel)
result_rate, result_data = wavfile.read(filename)

print(f"Result shape: {result_data.shape}")
print(f"Result data: {result_data}")
print(f"Expected shape: {data_2d_single_channel.shape}, Got shape: {result_data.shape}")
print(f"Shape preserved: {data_2d_single_channel.shape == result_data.shape}")

# Check if data values are preserved
if result_data.ndim == 1:
    data_equal = np.array_equal(result_data, data_2d_single_channel.flatten())
else:
    data_equal = np.array_equal(result_data, data_2d_single_channel)
print(f"Data values preserved: {data_equal}")

os.unlink(filename)

print("\n" + "=" * 60)
print("TEST 2: Compare 1-channel vs 2-channel behavior")
print("=" * 60)

# Test with 1 channel (2D array with shape (n, 1))
data_1ch = np.random.randint(-10000, 10000, size=(10, 1), dtype=np.int16)
print(f"1-channel input shape: {data_1ch.shape}")

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    filename = f.name

wavfile.write(filename, 8000, data_1ch)
_, result_1ch = wavfile.read(filename)
print(f"1-channel output shape: {result_1ch.shape}")
print(f"1-channel shape preserved: {data_1ch.shape == result_1ch.shape}")

os.unlink(filename)

# Test with 2 channels
data_2ch = np.random.randint(-10000, 10000, size=(10, 2), dtype=np.int16)
print(f"\n2-channel input shape: {data_2ch.shape}")

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    filename = f.name

wavfile.write(filename, 8000, data_2ch)
_, result_2ch = wavfile.read(filename)
print(f"2-channel output shape: {result_2ch.shape}")
print(f"2-channel shape preserved: {data_2ch.shape == result_2ch.shape}")

os.unlink(filename)

print("\n" + "=" * 60)
print("TEST 3: Test with 1D input for single channel")
print("=" * 60)

# Test what happens with 1D array input
data_1d = np.array([100, 200, 300], dtype=np.int16)
print(f"1D input shape: {data_1d.shape}")

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    filename = f.name

wavfile.write(filename, rate, data_1d)
_, result_1d = wavfile.read(filename)
print(f"1D output shape: {result_1d.shape}")
print(f"1D shape preserved: {data_1d.shape == result_1d.shape}")

os.unlink(filename)