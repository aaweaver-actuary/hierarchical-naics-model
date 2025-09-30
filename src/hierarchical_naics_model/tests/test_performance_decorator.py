import time
from functools import wraps


def log_test_performance(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Prefer test_run_id from kwargs, fallback to positional args if needed
        test_run_id = kwargs.get("test_run_id")
        if test_run_id is None:
            # Try to find test_run_id in positional args by name
            import inspect

            sig = inspect.signature(test_func)
            params = list(sig.parameters)
            if "test_run_id" in params:
                idx = params.index("test_run_id")
                if len(args) > idx:
                    test_run_id = args[idx]
        if test_run_id is None:
            test_run_id = "NO_RUN_ID"
        start = time.time()
        result = test_func(*args, **kwargs)
        end = time.time()
        duration = round(end - start, 1)
        log_line = (
            f"{test_run_id},{test_func.__module__},{test_func.__name__},"
            f"{start},{end},{duration}\n"
        )
        with open("test_performance.log", "a") as f:
            f.write(log_line)
        return result

    return wrapper
