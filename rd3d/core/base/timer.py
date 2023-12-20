import time
import torch

TIMER_ENABLE = True


class ScopeTimer:
    context = {}

    def __init__(self, title, average=False, verbose=True, enable=False):
        self.title = title
        self.is_verbose = verbose
        self.is_average = average
        self.enable = TIMER_ENABLE or enable

        self.duration = None
        self.tail = None
        if self.is_average and self.title not in self.context:
            self.context[self.title] = [0, 0]

    def __del__(self):
        pass

    def __enter__(self):
        if self.enable:
            torch.cuda.synchronize()
            self.t1 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            torch.cuda.synchronize()
            self.duration = (time.time() - self.t1) * 1e3
            if self.is_average:
                context = self.context[self.title]
                context[0] += self.duration  # accumulated duration
                context[1] += 1  # recording times
                if context[1] == 2:  # remove the unstable first measurement
                    context[0] = 2 * self.duration
                self.duration = context[0] / context[1]

            if self.is_verbose:
                content = '{}{:.3f}ms'.format(self.title, self.duration)
                content += f"({self.context[self.title][1]})" if self.is_average else ''
                content += (self.tail if self.tail is not None else '')
                print(content, flush=True)
        return exc_type is None


def measure_time(title=None, *args1, **kwargs1):
    def decorator(func):
        import functools

        @functools.wraps(func)
        def handler(*args2, **kwargs2):
            with ScopeTimer(title=f"{func.__name__}: " if title is None else title, *args1, *kwargs1):
                ret = func(*args2, **kwargs2)
            return ret

        return handler

    return decorator


