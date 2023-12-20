import torch
from utils import load_plugins


def demo():
    import argparse
    from pathlib import Path
    from rd3d import quick_demo

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path, required=True, help='the path of output')
    model, dataloader, cfg, args = quick_demo(parser)
    args.cfg = Path(args.cfg)
    return model, dataloader, args


def export_onnx():
    data_dict = dataloader.dataset[0]
    batch_dict = dataloader.dataset.collate_batch([data_dict])
    dataloader.dataset.load_data_to_gpu(batch_dict)
    batch_dict = dict(points=batch_dict['points'].view(1, -1, 5)[..., 1:].contiguous())
    output_path = fold / (file + '.onnx')
    with torch.no_grad():
        torch.onnx.export(model,
                          (batch_dict, {}),
                          output_path,
                          opset_version=11,
                          input_names=['points'],
                          output_names=['boxes', 'scores', 'nums'],
                          dynamic_axes={'points': {0: 'batch_size'},
                                        'boxes': {0: 'batch_size'},
                                        'scores': {0: 'batch_size'},
                                        'nums': {0: 'batch_size'}}
                          )
    return output_path


if __name__ == "__main__":
    model, dataloader, args = demo()
    model.cuda()
    model.eval()

    if args.onnx.suffix:
        fold = args.onnx.parent
        file = args.onnx.stem
    else:
        fold = args.onnx
        file = args.cfg.stem
    fold.mkdir(parents=True, exist_ok=True)

    save_path = export_onnx()
    print(f"save onnx model: {save_path}")
