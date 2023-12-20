from easydict import EasyDict


class Register(dict):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __setitem__(self, name, module):
        # if name in self:
        #     raise ValueError('%s exists' % name)
        super(Register, self).__setitem__(name, module)

    def register(self, *args):
        def register_to(module):
            for name in [module.__name__, *args]:
                self[name] = module
            return module

        return register_to

    def build_from_cfg(self, cfg=None, *args, **kwargs):
        name = None
        potential_name_field = ['NAME', 'name']
        for name_field in potential_name_field:
            name = cfg.get(name_field, name)
        if name is None:
            raise ValueError(f"one of the key {potential_name_field} is required in config dict")
        cfg = cfg if isinstance(cfg, EasyDict) else EasyDict(cfg)
        return self.build_from_name(name, cfg, *args, **kwargs)

    def build_from_name(self, name, *args, **kwargs):
        module = self[name]
        if isinstance(module, type):
            return module(*args, **kwargs)
        else:
            import functools
            return functools.partial(module, *args, **kwargs)
