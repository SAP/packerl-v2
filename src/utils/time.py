import signal
import sys
import time

from src.utils.proc import describe_exit, snapshot_process_and_kill


class TimedBlock:
    """
    A simple context manager to measure time spent in a with-block.
    """
    def __enter__(self):
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._dt = time.perf_counter() - self._t_start

    @property
    def elapsed_s(self) -> float:
        """
        :return: Time spent in the with-block, in seconds.
        """
        return self._dt

    @property
    def elapsed_ms(self) -> float:
        """
        :return: Time spent in the with-block, in milliseconds.
        """
        return self._dt * 1000


class TimeoutException(Exception):
    """
    A custom exception raised when a time-critical operation times out.
    """
    default_message = "TimeoutException: Operation took too long and was aborted."
    def __init__(self, message=None):
        self.message = message if message is not None else self.default_message
        super().__init__(self.message)


class TimeoutContext(object):
    """
    A light wrapper around context manager objects that allow for deliberate scope breaking
    and automated TimeoutException raises when too much time is spent in the scope.
    """
    class Break(Exception):
        """Break out of the with statement"""

    def __init__(self, sim, proc, shm_access_timeout=60):
        """
        :param sim: The context manager we are entering (ns3 'sim' object).
        :param proc: subprocess.Popen to inspect/kill on timeout.
        :param shm_access_timeout: Maximum time to wait for shared memory access, in seconds.
        """
        self.sim = sim
        self.proc = proc
        self.max_seconds = shm_access_timeout
        self._old_handler = None  # set/restore in enter/exit
        self.has_mem_access = False  # True while we have access to shared memory

    def _signal_handler(self, signum, frame):
        raise TimeoutException()

    def _on_timeout(self):
        # Runs in normal Python flow (safe to do I/O / kill), not in signal handler
        if self.proc is None:
            print("[timeout] No process to inspect/kill.", file=sys.stderr)
            return
        rc = self.proc.poll()
        if rc is None:
            print("[timeout] acquire timed out; ns-3 still running -> snapshot/kill", file=sys.stderr)
        else:
            print(f"[timeout] acquire timed out; ns-3 already exited: {describe_exit(rc)}", file=sys.stderr)
        try:
            snapshot_process_and_kill(self.proc, stream=sys.stderr)
        except Exception as e:
            print(f"[timeout] snapshot/kill failed: {e}", file=sys.stderr)

    def __enter__(self):

        # Fast-fail if ns-3 already died
        if self.proc is not None:
            rc = self.proc.poll()
            if rc is not None:
                raise RuntimeError(f"ns-3 process already exited: {describe_exit(rc)}")

        self._old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._signal_handler)
        signal.alarm(self.max_seconds)

        try:
            data = self.sim.__enter__()   # the only blocking call you care about
            self.has_mem_access = True
            return data
        except TimeoutException:
            self._on_timeout()
            raise
        finally:
            # Always disarm/restore even if __enter__ raised
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, self._old_handler)
            except Exception:
                pass

    def __exit__(self, etype, value, traceback):

        # Delegate cleanup to the wrapped context
        handled = self.sim.__exit__(etype, value, traceback)
        self.has_mem_access = False

        # Support deliberate break
        if etype == self.Break:
            return True

        return handled
