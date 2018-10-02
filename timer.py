
import timeit


class Timer:
    time_fn = timeit.default_timer

    def __enter__(self):
        self.start = self.time_fn()
        return self

    def __exit__(self, *args):
        self.end = self.time_fn()
        self.interval = self.end - self.start
