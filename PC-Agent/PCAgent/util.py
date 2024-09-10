import time
from functools import wraps

def print_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}' start executing:{start_time:.2f}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time  
        print(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds end:{end_time:.2f}")
        return result
    return wrapper