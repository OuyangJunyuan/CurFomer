import torch
import accelerate


def set_random_seed(seed, work_id=0, device_specific=True, work_specific=True):
    if work_specific:
        state = accelerate.accelerator.AcceleratorState()
        seed += work_id * state.num_processes

    accelerate.utils.set_seed(seed, device_specific=device_specific)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
