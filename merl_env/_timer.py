import time

class env_timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        if self._start_time is None:
            raise ValueError("Timer has not been started.")
        self._elapsed_time = time.time() - self._start_time
        self._start_time = None
        return self._elapsed_time

    def get_elapsed_time(self):
        if self._start_time is not None:
            # Timer is running
            return time.time() - self._start_time
        return self._elapsed_time