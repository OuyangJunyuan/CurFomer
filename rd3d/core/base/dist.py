import torch.distributed as dist
from accelerate.accelerator import PartialState

state = PartialState()


def recursively_apply(func, data, *args, **kwargs):
    from typing import Mapping
    construct = type(data)
    if isinstance(data, (tuple, list)):
        return construct(data, (recursively_apply(func, o, *args, **kwargs) for o in data))
    elif isinstance(data, Mapping):
        return construct({k: recursively_apply(func, v, *args, **kwargs) for k, v in data.items()})
    else:
        return func(data, *args, **kwargs)


def all_gather(data, on_main=False, opt="merge"):
    """

    Args:
        data: object
        on_main: return type(object)() if on_main otherwise return the gathered results.
        opt: how to combine the objects in list.

    Returns:

    """
    from typing import Mapping

    if not state.use_distributed:
        return data

    dest = not on_main or state.is_main_process

    gathered_list = [None] * state.num_processes if dest else None

    if on_main:
        dist.gather_object(data, gathered_list)
    else:
        dist.all_gather_object(gathered_list, data)

    if not dest:
        return None

    if isinstance(data, list):
        if opt == "merge":
            # [[0,1,2,3],[5,6,7,8]]->[0,1,2,3,4,5,6,7]
            return sum(gathered_list, [])
        elif opt == "insert":
            # [[0,2,4,6],[1,3,5,7]]->[[0,1],[2,3],[4,5],[6,7]]->[0,1,2,3,4,5,6,7]
            return sum([list(res) for res in zip(*gathered_list)], [])
        else:
            raise NotImplementedError
    elif isinstance(data, Mapping):
        return {k: d[k] for d in gathered_list for k in d}
    else:
        raise NotImplementedError


def all_reduce(tensor):
    def _all_reduce(data):
        dist.all_reduce(data)
        return data

    tensor = recursively_apply(lambda x: x if isinstance(x, torch.Tensor) else torch.Tensor([x]).cuda(), tensor)
    return recursively_apply(_all_reduce, tensor)


# if AcceleratorState().num_processes > 1:
#     import torch
#
#     torch.multiprocessing.set_sharing_strategy('file_system')

# class barrier:
#     def __init__(self):
#         self.num_processes = get_dist_state().num_processes
#
#     def __enter__(self):
#         if self.num_processes > 1:
#             dist.barrier()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.num_processes > 1:
#             dist.barrier()
#
#     def __call__(self, func):
#         return func
if state.num_processes > 1:
    import torch

    torch.multiprocessing.set_sharing_strategy('file_system')
