import os
import torch
import torch.nn as nn
from typing import Set
from .logger import create_logger, table_string

CKPT_PATTERN = 'checkpoint_epoch_'
logger = create_logger("ckpt", scene=True)


def __find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    try:
        import spconv.pytorch as spconv
    except:
        import spconv as spconv

    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(__find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def __load_state_dict(model: nn.Module, model_state_disk, *, strict=True):
    state_dict = model.state_dict()  # local cache of state_dict
    spconv_keys = __find_all_spconv_keys(model)

    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
            # with different spconv versions, we need to adapt weight shapes for spconv blocks
            # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

            val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
            if val_native.shape == state_dict[key].shape:
                val = val_native.contiguous()
            else:
                assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                if val_implicit.shape == state_dict[key].shape:
                    val = val_implicit.contiguous()

        if key in state_dict and state_dict[key].shape == val.shape:
            update_model_state[key] = val
            # logger.info('Update weight %s: %s' % (key, str(val.shape)))

    if strict:
        model.load_state_dict(update_model_state)
    else:
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)
    return state_dict, update_model_state


def load_from_file(filename, model=None, optimizer=None, scheduler=None, runner=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    device = torch.device('cpu') if model is None else next(model.parameters()).device

    checkpoint = torch.load(filename, map_location=device)
    logger.info(f'load checkpoint {filename} to {device}')

    if 'version' in checkpoint:
        logger.info(f'checkpoint trained from version: {checkpoint["version"]}')

    if model is not None:
        is_strict = optimizer is not None
        saved_state = checkpoint['model_state']
        state_dict, update_model_state = __load_state_dict(model, saved_state, strict=is_strict)

        missing_dict = {key: list(state_dict[key].shape) for key in state_dict if key not in update_model_state}
        unexpected_dict = {key: list(saved_state[key].shape) for key in saved_state if key not in state_dict}

        if missing_dict:
            logger.warning("\n%s" % table_string(missing_dict, header=['missing key', 'shape'], align="l"))
        if unexpected_dict:
            logger.warning("\n%s" % table_string(unexpected_dict, header=['unexpected key', 'shape'], align="l"))

        logger.info(f'loaded params for model ({len(update_model_state)}/{len(state_dict)})')

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        try:
            num_steps = scheduler.last_epoch
        except AttributeError:
            num_steps = scheduler.scheduler.last_epoch
        logger.info(f'loaded params for scheduler ({num_steps} steps)')

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f'loaded params for optimizer')

    if runner is not None:
        runner.load_state_dict(checkpoint['runner_state'])
        logger.info(f'loaded infos for runner at {runner.state} state'
                    f'({runner.cur_loops}l, {runner.cur_epochs}e, {runner.cur_iters}i)')


def __load_model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def __repo_version():
    try:
        from rd3d.version import __version__
        return __version__
    except:
        return 'none'


def save_to_file(model=None, optimizer=None, scheduler=None, runner=None, filename='checkpoint'):
    state = dict()
    if model is not None:
        model_state = model.state_dict()
        model_state = __load_model_state_to_cpu(model_state)
        state.update(model_state=model_state)
    try:
        from rd3d import version
        version = version.__version__
    except AttributeError:
        version = 'none'

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    runner_state = runner.state_dict() if runner is not None else {}
    state = dict(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        runner_state=runner_state,
        version=version,
    )
    torch.save(state, filename)


def potential_ckpts(ckpt, default=None):
    import glob
    from pathlib import Path
    if ckpt is not None:
        if Path(ckpt).is_dir():
            default = ckpt
        else:
            return [ckpt]
    assert not (ckpt is None and default is None)
    ckpt_list = glob.glob(str(default / f'*{CKPT_PATTERN}*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    return ckpt_list
