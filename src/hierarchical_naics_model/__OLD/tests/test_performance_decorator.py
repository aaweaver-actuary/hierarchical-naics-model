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


def test_log_test_performance_with_kwarg(tmp_path):
    # Patch open to write to tmp_path
    import builtins

    orig_open = builtins.open
    log_path = tmp_path / "test_performance.log"

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "test_performance.log":
            return orig_open(log_path, mode, *args, **kwargs)
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = fake_open

    @log_test_performance
    def dummy(test_run_id):
        return "ok"

    result = dummy(test_run_id="RUNID123")
    builtins.open = orig_open
    assert result == "ok"
    with open(log_path) as f:
        log = f.read()
    assert "RUNID123" in log


def test_log_test_performance_with_positional(tmp_path):
    import builtins

    orig_open = builtins.open
    log_path = tmp_path / "test_performance.log"

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "test_performance.log":
            return orig_open(log_path, mode, *args, **kwargs)
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = fake_open

    @log_test_performance
    def dummy(test_run_id):
        return "ok"

    result = dummy("RUNID456")
    builtins.open = orig_open
    assert result == "ok"
    with open(log_path) as f:
        log = f.read()
    assert "RUNID456" in log


def test_log_test_performance_no_run_id(tmp_path):
    import builtins

    orig_open = builtins.open
    log_path = tmp_path / "test_performance.log"

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "test_performance.log":
            return orig_open(log_path, mode, *args, **kwargs)
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = fake_open

    @log_test_performance
    def dummy():
        return "ok"

    result = dummy()
    builtins.open = orig_open
    assert result == "ok"
    with open(log_path) as f:
        log = f.read()
    assert "NO_RUN_ID" in log
