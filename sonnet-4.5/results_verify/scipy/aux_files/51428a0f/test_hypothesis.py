from hypothesis import given, strategies as st
import numpy as np

def face_logic(face_array, gray):
    if gray is True:
        return (0.21 * face_array[:, :, 0] + 0.71 * face_array[:, :, 1] +
                0.07 * face_array[:, :, 2]).astype('uint8')
    return face_array

@given(st.integers(min_value=1, max_value=10))
def test_face_gray_truthy_values(val):
    mock_face = np.random.randint(0, 256, size=(768, 1024, 3), dtype='uint8')
    result = face_logic(mock_face, gray=val)

    if val:
        assert result.ndim == 2, \
            f"Truthy value {val} should trigger grayscale conversion but returned shape {result.shape}"

if __name__ == "__main__":
    test_face_gray_truthy_values()