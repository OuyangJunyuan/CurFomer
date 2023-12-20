class AttributePath:
    def __init__(self, attrs=None):
        self._path = attrs or []
        self._value = None
        self._index = 0

    @property
    def value(self):
        return self._value

    def __add__(self, other: str):
        return AttributePath(self._path + [other])

    def __getattr__(self, item):
        return self + item

    def __getitem__(self, item):
        self._index = item
        return self

    def __call__(self, value):
        self._value = value
        return self

    def __iter__(self):
        yield from self._path

    def get_state(self, *args, **kwargs):
        if self._index < len(args):
            arg = args[self._index]
        else:
            arg = list(kwargs.values())[len(args) - self._index]
        for name in self:
            arg = getattr(arg, name)
        return arg

    def __str__(self):
        return '.'.join(self._path) + ":=%s" % self._value


def dispatch(func):
    from functools import wraps
    from collections import defaultdict

    assert func.__annotations__
    dispatch.cache = getattr(dispatch, 'cache', defaultdict(dict))
    cache_key = func.__module__ + func.__name__
    infos = [v[func.__code__.co_varnames.index(k)]
             for k, v in func.__annotations__.items()
             if isinstance(v, AttributePath)]
    func.__dispatch_infos__ = infos
    dispatch.cache[cache_key][tuple(info.value for info in infos)] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        current_state = tuple(info.get_state(*args, **kwargs) for info in func.__dispatch_infos__)
        impl = dispatch.cache[cache_key].get(current_state, lambda *_, **__: None)
        return impl(*args, **kwargs)

    return wrapper


when = AttributePath()
