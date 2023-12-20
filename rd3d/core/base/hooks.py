from easydict import EasyDict
from .register import Register

HOOKS = Register("hooks")


class Hook:
    _hooks = []

    @staticmethod
    def insert(hook):
        def find_place_to_insert_by_priority():
            for i, h in enumerate(Hook._hooks):
                if hook.priority <= h.priority:
                    return i
            return len(Hook._hooks)

        hook.priority = getattr(hook, 'priority', 50)
        Hook._hooks.insert(find_place_to_insert_by_priority(), hook)
        return hook

    @staticmethod
    def begin_and_end(func):
        from functools import wraps

        func_name = func.__name__
        func_begin = 'before_%s' % func_name
        func_end = 'after_%s' % func_name

        def valid(hook, name):
            return getattr(hook, 'enable', True) and hasattr(hook, name)

        @wraps(func)
        def wrapper(*args, **kwargs):

            for hook in Hook._hooks:
                if valid(hook, func_begin):
                    getattr(hook, func_begin)(*args, **kwargs)

            ret = list(func(*args, **kwargs) or [])

            for hook in Hook._hooks:
                if valid(hook, func_end):
                    hook.__setattr__('%s_output' % func_name, ret)
                    getattr(hook, func_end)(*args, **kwargs)

            return ret

        return wrapper

    @staticmethod
    def insert_by_configs(cfgs):
        def get_path(name_field):
            paths = name_field.split('.')
            *paths, hook_name = paths
            return '.'.join(paths), hook_name

        for cfg in cfgs:
            path, cfg['name'] = get_path(cfg['name'])
            if path:
                __import__(path)
            Hook.insert(HOOKS.build_from_cfg(cfg))

    @classmethod
    def info(cls):
        from ..logger import table_string
        infos = {type(v).__name__: v.priority for v in cls._hooks}
        return table_string(infos, header=['hook', 'priority'])
