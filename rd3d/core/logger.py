from accelerate import logging as warp

logging = warp.logging


def create_logger(name=None, file=None, scene=True, level=logging.INFO):
    def get_formatter():
        return "[%(asctime)s %(name)s %(levelname)s] %(message)s"

    def stream_handler():
        import sys
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    def file_handler():
        from pathlib import Path
        fold = Path(file).parent
        if not fold.exists():
            fold.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(filename=file)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    exist = name is None or name in logging.Logger.manager.loggerDict
    warped_logger = warp.get_logger(name)
    logger = warped_logger.logger
    logger.setLevel(level)
    logger.propagate = False  # disable the message propagation to the root logger

    if not exist:
        formatter = logging.Formatter(get_formatter())
        if scene:
            logger.addHandler(stream_handler())
        if file:
            logger.addHandler(file_handler())

    return warped_logger


def table_string(data_dict, title=None, header=None, exclude=None, width=0, **kwargs):
    def iterable(x):
        from typing import Iterable, ByteString
        return isinstance(x, Iterable) and not isinstance(x, (str, ByteString))

    def make_string(data, _title=None, _header=None, _deep=0, _width=50):
        import textwrap
        import prettytable
        from typing import Iterable, Sequence, Mapping

        if iterable(data):
            kvs = list(data.items() if isinstance(data, Mapping) else enumerate(data))
            if _title or isinstance(data, Mapping) or sum([isinstance(v, Iterable) for _, v in kvs]):
                tb = prettytable.PrettyTable(title=_title, field_names=header, header=_header is not None, **kwargs)
                tb.set_style(prettytable.SINGLE_BORDER)
                tb.add_rows([[k, make_string(data=v, _deep=_deep + 1, _width=_width)] for k, v in kvs])
                return tb.get_string(fields=tb.field_names[1:], border=False) \
                    if isinstance(data, Sequence) and _deep else tb.get_string()
        return textwrap.fill(data.__str__(), width=_width) if _width else data.__str__()

    if exclude:
        data_dict = {k: v for k, v in data_dict.items() if k not in exclude}

    msg = make_string(data_dict, _title=title, _header=header, _width=width)
    return msg


def dict_string(data_dict, exclude=None):
    def make_string(data_dict):
        from rich.pretty import pretty_repr
        ans = pretty_repr(data_dict, expand_all=False)
        return ans

    if exclude:
        data_dict = {k: v for k, v in data_dict.items() if k not in exclude}

    msg = make_string(data_dict)
    return msg
