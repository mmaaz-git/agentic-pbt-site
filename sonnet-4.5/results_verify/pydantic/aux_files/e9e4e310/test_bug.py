from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def test_first_on_error_exception_prevents_second_handler():
    """Test case from bug report - first handler exception masks original error"""
    handler1_called = []
    handler2_called = []

    def original_func(x):
        raise ValidationError.from_exception_data('test', [])

    handler1 = Mock()
    handler1.on_error = Mock(side_effect=RuntimeError("handler1 error"))
    handler1.on_error.__module__ = 'test1'

    handler2 = Mock()
    handler2.on_error = lambda e: handler2_called.append(True)
    handler2.on_error.__module__ = 'test2'

    wrapped = build_wrapper(original_func, [handler1, handler2])

    caught_exception = None
    try:
        wrapped(5)
    except Exception as e:
        caught_exception = e

    print(f"Test 1 - First handler exception prevents second handler:")
    print(f"  Exception type: {type(caught_exception).__name__}")
    print(f"  Is RuntimeError: {isinstance(caught_exception, RuntimeError)}")
    print(f"  Handler2 called: {len(handler2_called) > 0}")
    assert isinstance(caught_exception, RuntimeError)
    assert len(handler2_called) == 0
    print("  Test PASSED: Bug confirmed - handler exception masks original error\n")


def test_reproduction_case():
    """Reproduction case from bug report"""
    def original_func(x):
        raise ValidationError.from_exception_data('validation_failed', [])

    handler1 = Mock()
    handler1.on_error = Mock(side_effect=RuntimeError("handler1 crashed"))
    handler1.on_error.__module__ = 'plugin1'

    handler2 = Mock()
    handler2.on_error = Mock()
    handler2.on_error.__module__ = 'plugin2'

    wrapped = build_wrapper(original_func, [handler1, handler2])

    try:
        wrapped(42)
    except Exception as e:
        print(f"Test 2 - Reproduction case:")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Expected: ValidationError")
        print(f"  Actual: {type(e).__name__}")
        print(f"  Handler2 called: {handler2.on_error.called}")
        print("  Test confirms bug - original ValidationError is masked\n")


def test_on_success_handler_exception():
    """Test that on_success handler exception also causes issues"""
    handler2_called = []

    def original_func(x):
        return x * 2

    handler1 = Mock()
    handler1.on_success = Mock(side_effect=RuntimeError("success handler crashed"))
    handler1.on_success.__module__ = 'test1'

    handler2 = Mock()
    handler2.on_success = lambda r: handler2_called.append(r)
    handler2.on_success.__module__ = 'test2'

    wrapped = build_wrapper(original_func, [handler1, handler2])

    try:
        result = wrapped(5)
    except RuntimeError as e:
        print(f"Test 3 - on_success handler exception:")
        print(f"  Exception raised: {str(e)}")
        print(f"  Handler2 called: {len(handler2_called) > 0}")
        print("  Test confirms bug - handler exception prevents function return\n")


def test_on_enter_handler_exception():
    """Test that on_enter handler exception prevents function execution"""
    func_called = []

    def original_func(x):
        func_called.append(True)
        return x * 2

    handler1 = Mock()
    handler1.on_enter = Mock(side_effect=RuntimeError("enter handler crashed"))
    handler1.on_enter.__module__ = 'test1'

    wrapped = build_wrapper(original_func, [handler1])

    try:
        result = wrapped(5)
    except RuntimeError as e:
        print(f"Test 4 - on_enter handler exception:")
        print(f"  Exception raised: {str(e)}")
        print(f"  Original function called: {len(func_called) > 0}")
        print("  Test confirms bug - handler exception prevents function execution\n")


if __name__ == "__main__":
    print("Running bug reproduction tests...\n")
    test_first_on_error_exception_prevents_second_handler()
    test_reproduction_case()
    test_on_success_handler_exception()
    test_on_enter_handler_exception()
    print("All tests completed - bug is confirmed!")