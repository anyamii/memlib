"""Bounded daemon worker: caller returns on deadline even if a source hangs.

One outstanding call per boundary; a hung worker makes later calls fail closed.
Threads cannot be killed: isolate untrusted/native calls in processes in deployments.
"""
import queue
import threading
import time

class Deadline:
    def __init__(self):
        self.lock = threading.Lock()

    def call(self, function, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.lock.acquire(blocking=False):
            raise TimeoutError
        result = queue.Queue(maxsize=1)
        def run():
            try:
                outcome = (True, function())
            except BaseException as exc:
                outcome = (False, exc)
            finally:
                self.lock.release()
            result.put(outcome)
        threading.Thread(target=run, daemon=True).start()
        try:
            ok, value = result.get(timeout=max(0, deadline-time.monotonic()))
        except queue.Empty:
            raise TimeoutError from None
        if not ok:
            raise value
        return value
