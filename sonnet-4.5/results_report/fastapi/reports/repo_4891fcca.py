from fastapi.dependencies.utils import get_dependant

# Test case: Using Python keyword "if" as a string type annotation
def endpoint_function(x: "if"):
    """Function with a keyword as string type annotation"""
    return {"x": x}

# Try to process this function through FastAPI's dependency system
try:
    dependant = get_dependant(path="/test", call=endpoint_function)
    print("SUCCESS: Function processed without error")
    print(f"Dependant: {dependant}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()