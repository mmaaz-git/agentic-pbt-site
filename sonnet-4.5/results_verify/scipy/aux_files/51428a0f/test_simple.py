import numpy as np

def face_logic(face_array, gray):
    if gray is True:
        face = (0.21 * face_array[:, :, 0] + 0.71 * face_array[:, :, 1] +
                0.07 * face_array[:, :, 2]).astype('uint8')
        return face
    return face_array

mock_face = np.random.randint(0, 256, size=(768, 1024, 3), dtype='uint8')

result = face_logic(mock_face, gray=1)
print(f"face(gray=1) shape: {result.shape}")
print(f"Expected: (768, 1024) for grayscale")
print(f"Actual: {result.shape} (color image)")

# Also test with gray=True for comparison
result_true = face_logic(mock_face, gray=True)
print(f"\nface(gray=True) shape: {result_true.shape}")
print(f"This works correctly, produces grayscale with shape (768, 1024)")