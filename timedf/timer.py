import time
import logging


logger = logging.getLogger(__name__)


VERBOSITY_LEVELS = (0, 1, 2, 3)


class TimerManager:
    """
    Utility timer that can measure time using `timeit` function.

    Intended use is through context manager like.

    Notes
    ------
    TimeManager supports nested timings if called through the same object.
    If you give the same name to measure twice, this will either lead to an error,
    if allow_acc is False or accumulate measurements.

    Examples
    ----------
    >>> tm = TimerManager
    >>> with tm.timeit('heavy_call'):
    >>>     heavy_call()
    """

    def __init__(self, allow_acc=True, verbosity: int = 0) -> None:
        """Initialize root timer.

        Parameters
        ----------
        allow_acc, optional
            Allow accumulating of measured time, by default True

        verbosity:
            Write timer stack status to stdout.
            Possible values:
                0: no writing (default)
                1: write about exit only
                2: write about exit and enter
        """
        # name for the next timer to start, also acts as timer state
        self.prepared_name = None
        self.allow_acc = allow_acc
        self.timer_stack = self.TimerStack(allow_acc=allow_acc)
        self.verbosity = verbosity
        self.profiles = {}

    def reset(self):
        """Reset timer state"""
        self.prepared_name = None
        self.timer_stack = self.TimerStack(allow_acc=self.allow_acc)

    @staticmethod
    def check_verbosity(verbosity):
        if verbosity not in VERBOSITY_LEVELS:
            raise ValueError(
                f"Provided verbosity={verbosity}," "but possible values are {VERBOSITY_LEVELS}"
            )

    @property
    def verbosity(self):
        return self._verbose

    @verbosity.setter
    def verbosity(self, value):
        self.check_verbosity(value)
        self._verbosity = value

    def timeit(self, name):
        if self.prepared_name is not None:
            raise ValueError(f'Unfinished timer named "{name}" discovered')

        self.prepared_name = name
        return self

    def __enter__(self):
        if self.prepared_name is None:
            raise ValueError("Attempted to start timer, but it has no name")

        self.timer_stack.push(self.prepared_name)
        if self._verbosity > 1:
            level = self.timer_stack.get_current_level() - 1
            print("  " * level + f"{self.timer_stack.get_full_name()} started")
        if self._verbosity > 2:
            import cProfile

            profile = cProfile.Profile()
            profile.enable()
            self.profiles[self.timer_stack.get_full_name()] = profile
        self.prepared_name = None
        return self

    def __exit__(self, type, value, traceback):
        fullname = self.timer_stack.get_full_name()
        self.timer_stack.pop()
        if self._verbosity > 0:
            level = self.timer_stack.get_current_level()
            print("  " * level + f"{fullname}: {self.timer_stack.fullname2time[fullname]}")
        if self._verbosity > 2:
            from pstats import SortKey

            profile = self.profiles.pop(fullname)
            profile.disable()
            profile.print_stats(SortKey.CUMULATIVE)

    def get_results(self):
        return self.timer_stack.get_results()

    class TimerStack:
        """Keeps internal stack of running timers (time and name) and resulting report."""

        SEPARATOR = "."

        def __init__(self, allow_acc=False) -> None:
            self.name_stack = []
            self.start_stack = []

            self.allow_acc = allow_acc
            self.fullname2time = {}

        def push(self, name):
            self._check_name(name)
            self.start_stack.append(time.perf_counter())
            self.name_stack.append(name)

        def pop(self):
            fullname = self.get_full_name()
            self.name_stack.pop()

            self._check_overwrite(fullname)
            delta = time.perf_counter() - self.start_stack.pop()
            self.fullname2time[fullname] = self.fullname2time.get(fullname, 0) + delta

        def _check_name(self, name):
            if self.SEPARATOR in name:
                raise ValueError(
                    f'Provided name: "{name}" contains separator symbols "{self.SEPARATOR}"'
                )

        def _check_overwrite(self, fullname):
            if not self.allow_acc and fullname in self.fullname2time:
                raise ValueError(f"Trying to rewrite measurment for {fullname}")

        def get_full_name(self):
            return self.SEPARATOR.join(self.name_stack)

        def get_current_level(self):
            return len(self.start_stack)

        def get_results(self):
            return dict(self.fullname2time)


# # Global timer manager for benchmarks
tm = TimerManager()
