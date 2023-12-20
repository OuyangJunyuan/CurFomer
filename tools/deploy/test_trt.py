from rd3d import quick_demo
from rd3d.hooks.eval import Latency

latency = Latency()


def evaluate(args, engine_file, dataloader):
    import torch
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt
    import pycuda.autoinit
    from tqdm import tqdm
    from utils import load_plugins

    logger = trt.Logger(trt.Logger.ERROR)
    # create engine
    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    bs = dataloader.batch_size
    points_index = engine.get_binding_index("points")
    num_points = engine.get_binding_shape(points_index)[1]
    pred_labels = []
    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        context.set_binding_shape(points_index, (bs, num_points, 4))
        assert context.all_binding_shapes_specified

        h_inputs = {'points': np.zeros((bs, num_points, 4), dtype=float)}
        d_inputs = {}
        h_outputs = {}
        d_outputs = {}
        t_outputs = {}
        for binding in engine:
            if engine.binding_is_input(binding):
                d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
            else:
                size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)
        bar = tqdm(iterable=dataloader)
        for batch_dict in bar:
            dataloader.dataset.load_data_to_gpu(batch_dict)
            if bs != batch_dict['batch_size']:
                bs = batch_dict['batch_size']
                context.set_binding_shape(engine.get_binding_index("points"), (bs, num_points, 4))
                assert context.all_binding_shapes_specified

                h_inputs = {'points': np.zeros((bs, num_points, 4), dtype=float)}
                d_inputs = {}
                h_outputs = {}
                d_outputs = {}
                t_outputs = {}
                for binding in engine:
                    if engine.binding_is_input(binding):
                        d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
                    else:
                        size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
                        dtype = trt.nptype(engine.get_binding_dtype(binding))
                        h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                        d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)

            h_inputs = {'points': batch_dict['points'].view(bs, -1, 5)[..., 1:].contiguous().cpu().numpy()}
            for key in h_inputs:
                cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
            latency.add_batch_before()
            context.execute_async_v2(
                bindings=[int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs],
                stream_handle=stream.handle)
            for key in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
            stream.synchronize()
            latency.add_batch_after()
            rt = f"{latency.val:.2f}({latency.avg:.2f})"
            bar.set_postfix({'runtime': rt})

            output_nums = torch.from_numpy(h_outputs['nums']).cuda()
            output_scores = torch.from_numpy(h_outputs['scores'].reshape((bs, -1))).cuda()
            output_boxes = torch.from_numpy(h_outputs['boxes'].reshape((bs, output_scores.shape[-1], -1))).cuda()
            if args.viz:
                from rd3d.utils import viz_utils
                viz_utils.viz_scene(points=batch_dict['points'].view(bs, -1, 5)[0, 1:],
                                    boxes=output_boxes[0, :output_nums[0], :7])

            final_output_dicts = [{'pred_boxes': output_boxes[i, :output_nums[i], :7],
                                   'pred_labels': output_boxes[i, :output_nums[i], -1].int(),
                                   'pred_scores': output_scores[i, :output_nums[i]]}
                                  for i in range(bs)]

            pred_labels.extend(dataloader.dataset.generate_prediction_dicts(batch_dict, final_output_dicts,
                                                                            dataloader.dataset.class_names))
    return pred_labels


def main():
    import pickle
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--cache', type=Path, default=Path("/home/nrsl/workspace/temp/RobDet3D/cache/result.pkl"))
    _, dataloader, cfg, args = quick_demo(parser)

    engine_file = args.engine
    eval_file = args.cache

    if eval_file.exists():
        pred_labels = pickle.load(open(eval_file, 'br'))
    else:
        pred_labels = evaluate(args, engine_file, dataloader)
        pickle.dump(pred_labels, open(eval_file, 'wb'))

    result_str, result_dict = dataloader.dataset.evaluation(pred_labels, dataloader.dataset.class_names)
    print(result_str)


if __name__ == "__main__":
    main()
