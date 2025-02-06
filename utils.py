import socket
import time
import functools
import threading
def retry_transmission_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self_or_cls = args[0] if args else None
        is_method = hasattr(self_or_cls, '__class__')
        i=0
        while i<10:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    if is_method:
                        result = func(self_or_cls,client_socket,*args[1:], **kwargs)
                    else:
                        result = func(client_socket,*args, **kwargs)
                    break
            except Exception as e:
                print(e)
                time.sleep(0.1)
            i+=1
    return wrapper

def threaded(func):
    """Decorator to run a function in a new thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread  # Return the thread in case the caller wants to manage it
    return wrapper