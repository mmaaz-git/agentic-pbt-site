#!/usr/bin/env python3

# Test 1: Basic reproduction as shown in bug report
import Cython.Compiler.TypeInference as TI
import Cython.Compiler.PyrexTypes as PT

print("=== Test 1: Basic reproduction ===")
t1 = PT.c_int_type
t2 = PT.error_type

result_12 = TI.find_spanning_type(t1, t2)
result_21 = TI.find_spanning_type(t2, t1)

print(f'find_spanning_type(int, error) = {result_12}')
print(f'find_spanning_type(error, int) = {result_21}')
print(f'Equal? {result_12 == result_21}')
print()

# Test 2: Let's test with other types as well
print("=== Test 2: Testing with other types ===")
types_to_test = [
    ('c_int_type', PT.c_int_type),
    ('c_double_type', PT.c_double_type),
    ('c_float_type', PT.c_float_type),
    ('py_object_type', PT.py_object_type),
    ('c_char_ptr_type', PT.c_char_ptr_type),
    ('c_void_type', PT.c_void_type),
]

for type_name, type_obj in types_to_test:
    result_te = TI.find_spanning_type(type_obj, PT.error_type)
    result_et = TI.find_spanning_type(PT.error_type, type_obj)
    if result_te != result_et:
        print(f'{type_name}: find_spanning_type(type, error) = {result_te}, find_spanning_type(error, type) = {result_et} [NOT EQUAL]')
    else:
        print(f'{type_name}: Both directions give {result_te} [OK]')
print()

# Test 3: Run the hypothesis test
print("=== Test 3: Hypothesis test ===")
try:
    from hypothesis import given, strategies as st, settings

    all_type_objects = []
    for name in dir(PT):
        if name.endswith('_type') and not name.startswith('_'):
            obj = getattr(PT, name)
            if hasattr(obj, '__class__') and not callable(obj) and not isinstance(obj, dict):
                if hasattr(obj, 'is_int') or hasattr(obj, 'is_error'):
                    all_type_objects.append(obj)

    type_strategy = st.sampled_from(all_type_objects)

    @given(type_strategy, type_strategy)
    @settings(max_examples=1000)
    def test_find_spanning_type_commutativity(t1, t2):
        result_forward = TI.find_spanning_type(t1, t2)
        result_backward = TI.find_spanning_type(t2, t1)
        assert result_forward == result_backward, f"Failed for {t1} and {t2}: {result_forward} != {result_backward}"

    test_find_spanning_type_commutativity()
    print("Hypothesis test passed!")

except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Could not run hypothesis test: {e}")

# Test 4: Test the underlying issue - assignable_from behavior
print()
print("=== Test 4: Understanding assignable_from behavior ===")
print(f"error_type.assignable_from(c_int_type): {PT.error_type.assignable_from(PT.c_int_type)}")
print(f"c_int_type.assignable_from(error_type): {PT.c_int_type.assignable_from(PT.error_type)}")
print(f"error_type.assignable_from(py_object_type): {PT.error_type.assignable_from(PT.py_object_type)}")
print(f"py_object_type.assignable_from(error_type): {PT.py_object_type.assignable_from(PT.error_type)}")

# Test 5: Check same_as_resolved_type for ErrorType
print()
print("=== Test 5: ErrorType's same_as_resolved_type behavior ===")
print(f"error_type.same_as_resolved_type(c_int_type): {PT.error_type.same_as_resolved_type(PT.c_int_type)}")
print(f"error_type.same_as_resolved_type(py_object_type): {PT.error_type.same_as_resolved_type(PT.py_object_type)}")
print(f"error_type.same_as_resolved_type(error_type): {PT.error_type.same_as_resolved_type(PT.error_type)}")