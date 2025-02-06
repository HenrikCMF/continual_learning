import socket
import time
import functools
import threading
def retry_until_success(func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    result = func(client_socket,*args, **kwargs)
            except Exception as e:
                time.sleep(0.1)
    return wrapper

def threaded(func):
    """Decorator to run a function in a new thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread  # Return the thread in case the caller wants to manage it
    return wrapper