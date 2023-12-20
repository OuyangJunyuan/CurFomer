import torch
import numpy as np
from easydict import EasyDict

def get_matching_boxes(boxes, name='euclidean', discard=0, radius=1.0):
    """
    Args:
        boxes: [n,9]
        name: name of cluster method, can be one of the 'euclidean', 'dbscan', or 'iou'
        discard: cluster with boxes less than discard are ignored.
        radius: neighbour search radius for 'euclidean', 'dbscan' or iou threshold for 'iou' method.
    Returns:
        matches: list([k_i,9])
    """
    from scipy import stats, spatial

    if name == 'euclidean':  # fastest
        tree = spatial.cKDTree(boxes[:, :3])
        qbp_boxes = tree.query_ball_point(boxes[:, :3], r=radius)
        qbp_boxes_filt = [tuple(sets) for sets in qbp_boxes if len(sets) >= discard]
        return [boxes[indices, ...] for indices in list(set(qbp_boxes_filt))]
    elif name == 'dbscan':  # moderate
        from sklearn import cluster
        label = cluster.DBSCAN(eps=radius, min_samples=discard).fit(boxes[:, :3]).labels_
        return [boxes[np.where(label == i)[0]] for i in range(0, label.max() + 1)]
    elif name == 'iou':  # slowest
        from ..ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
        iou_threshold = radius
        index = list(range(len(boxes)))
        sort_index = np.argsort(boxes[:, 7])
        boxes = boxes[sort_index]
        boxes_gpu = torch.from_numpy(boxes[:, :7]).cuda()
        iou = boxes_iou3d_gpu(boxes_gpu, boxes_gpu).cpu()

        matches = []
        states = np.ones(len(boxes), bool)
        for i in index:
            if states[i]:
                select = torch.nonzero(iou[i] > iou_threshold)[:, 0].cpu()
                matches.append(boxes[select, :].reshape(-1, boxes.shape[-1]))
                states[select] = False
        return [m for m in matches if len(m) >= discard]
    else:
        raise NotImplementedError


def kde_box_fusion_one_cluster(boxes, weights, yaw_opt=True,
                               bw_loc=1.0, bw_size=2.0, bw_yaw=0.1, bw_score=2.0, bw_label=0.5):
    """
    Notes: should take better method to fuse yaws to avoid combining x and x+180d
    Args:
        boxes: [n,(x,y,z,l,w,h,yaw,score,label)]
        weights: [n]

    Returns:
        boxes: (x,y,z,l,w,h,yaw,score,label)
    """

    def get_kde(x, w, bw):
        """
        Notes: a bw of None means using adaptive scott bw.
        Args:
            x: [d,n] data points
            w: [n] weights
            bw: float

        Returns: kde model, test by kde(estimate point)

        """

        return stats.gaussian_kde(x, bw_method=bw, weights=w)

    def get_kde_estimation(x, w, bw, grid_sample=False, fallback_to_mean=True):
        """
        Notes: by default, we do not perform grid sample to search probability peek and do not fall back to mean.
        """

        x = x[None, ...] if len(x.shape) == 1 else x.T
        try:
            kde = get_kde(x, w, bw)
            if grid_sample:
                if x.shape[0] == 1:
                    estimate_points = np.mgrid[
                                      min(x[0, :]):max(x[0, :]):100j]
                elif x.shape[0] == 2:
                    estimate_points = np.mgrid[
                                      min(x[0, :]):max(x[0, :]):20j,
                                      min(x[1, :]):max(x[1, :]):20j]
                elif x.shape[0] == 3:
                    estimate_points = np.mgrid[
                                      min(x[0, :]):max(x[0, :]):20j,
                                      min(x[1, :]):max(x[1, :]):20j,
                                      min(x[2, :]):max(x[2, :]):20j]
                else:
                    # using random test point ?
                    raise NotImplementedError
                estimate_points = np.vstack(estimate_points).reshape(x.shape[0], -1)
                estimate_values = kde(estimate_points)
                max_estimate_value_point = estimate_points[:, estimate_values.argmax(axis=-1)]
                return max_estimate_value_point
            else:
                estimate_values = kde(x)
                max_estimate_value_point = x[:, estimate_values.argmax(axis=-1)]
                return max_estimate_value_point

        except:
            # print("kde fails %f!" % bw)
            if fallback_to_mean:
                # fail to estimate value by kde manner.
                return np.average(x, axis=-1, weights=w)
            else:
                return None

    def kde_location_fusion():
        new_locs = get_kde_estimation(locs, weights, bw_loc, fallback_to_mean=False)  # D-KDE or fallback to 2D+1D
        if new_locs is None:
            new_xy = get_kde_estimation(locs[:, :2], weights, bw_loc)
            new_z = get_kde_estimation(locs[:, 2:3], weights, bw_loc)
            new_locs = np.hstack((new_xy, new_z))
        return new_locs

    def kde_size_fusion():
        new_l = get_kde_estimation(sizes[:, 0:1], weights, bw_size, grid_sample=True)
        new_w = get_kde_estimation(sizes[:, 1:2], weights, bw_size, grid_sample=True)
        new_h = get_kde_estimation(sizes[:, 2:3], weights, bw_size, grid_sample=True)
        new_sizes = np.hstack((new_l, new_w, new_h))
        return new_sizes

    def kde_yaw_fusion():
        def flip_opposite_yaw(raw_yaws):
            def angle_diff(a1, a2, period=2 * np.pi):
                val = a1 - a2
                return val - np.floor(val / period + 0.5) * period  # [-pi,pi)

            def get_kde_estimation_radian(x, q, weights, bandwidth):
                import torch

                kernel = torch.distributions.Normal(loc=0., scale=1.)
                x, q = x[None, :], q[:, None]
                sample = angle_diff(q, x) / bandwidth
                prob = (weights[None, :] * kernel.log_prob(torch.from_numpy(sample)).exp().numpy()).sum(axis=-1)
                return prob

            prim_dir = raw_yaws[get_kde_estimation_radian(raw_yaws, raw_yaws, weights, 0.1).argmax()]
            dist = angle_diff(raw_yaws, prim_dir)
            flip_flag = np.abs(dist) > np.pi / 2
            raw_yaws[flip_flag] = (dist - np.sign(dist) * np.pi)[flip_flag]
            return raw_yaws

        yaw_vectors = np.hstack((np.cos(yaws), np.sin(yaws)))
        new_yaw_vectors = get_kde_estimation(yaw_vectors, weights, bw_yaw, fallback_to_mean=False)
        if new_yaw_vectors is None:
            if yaw_opt:
                yaws[:, 0] = flip_opposite_yaw(yaws.ravel())
                yaw_vectors = np.hstack((np.cos(yaws), np.sin(yaws)))
            mean_yaw_vectors = np.average(yaw_vectors, axis=0, weights=weights)
            error = np.linalg.norm(yaw_vectors - mean_yaw_vectors)
            new_yaw_vectors = yaw_vectors[np.argmin(error, axis=0)]
        new_yaws = np.arctan2(new_yaw_vectors[1], new_yaw_vectors[0])
        return new_yaws

    def kde_score_fusion():
        new_scores = get_kde_estimation(scores, weights, bw_score, grid_sample=True)
        return new_scores

    def kde_label_fusion():
        new_labels = get_kde_estimation(labels, weights, bw_label, fallback_to_mean=False)
        if new_labels is None:
            unique, counts = np.unique(labels, return_counts=True)
            new_labels = int(unique[np.argmax(counts)])
        return new_labels

    # remove duplicated boxes caused by randomly disabled tta items.
    boxes, indices = np.unique(boxes, axis=0, return_index=True)
    weights = weights[indices]
    locs, sizes, yaws, scores, labels, res = np.split(boxes, (3, 6, 7, 8, 9), axis=-1)
    assert res.shape[-1] == 0

    new_boxes = np.hstack(
        (
            kde_location_fusion(),
            kde_size_fusion(),
            kde_yaw_fusion(),
            kde_score_fusion(),
            kde_label_fusion(),
        )
    )
    return new_boxes


def kde_boxes_fusion(boxes, kde_config):
    bw_config = kde_config['BANDWIDTH']
    match_config = kde_config['MATCH']
    new_boxes = [kde_box_fusion_one_cluster(cluster, cluster[:, 7], **bw_config)[None, :]
                 for cluster in get_matching_boxes(boxes, **match_config)]
    if new_boxes:
        new_boxes = np.vstack(new_boxes)  # list(n,[9]) -> (n,9)
    else:
        new_boxes = np.zeros([0, boxes.shape[-1]])
    return new_boxes


def kde_boxes_fusion_tta(pred_dict, kde_config):
    bw_config = kde_config['BANDWIDTH']
    match_config = kde_config['MATCH']
    boxes = torch.hstack((pred_dict['pred_boxes'],
                          pred_dict['pred_scores'][:, None],
                          pred_dict['pred_labels'][:, None]))
    boxes_np = boxes.cpu().numpy()
    new_boxes = [kde_box_fusion_one_cluster(cluster, cluster[:, 7], **bw_config)[None, :]
                 for cluster in get_matching_boxes(boxes_np, **match_config)]
    if new_boxes:
        new_boxes = np.vstack(new_boxes)
        new_boxes = boxes.new_tensor(new_boxes)
        final_boxes = new_boxes[:, :7]
        final_scores = new_boxes[:, 7]
        final_labels = new_boxes[:, 8].long()
        return final_boxes, final_scores, final_labels
    else:
        return pred_dict['pred_boxes'], pred_dict['pred_scores'], pred_dict['pred_labels']


def nms_boxes_fusion_tta(pred_dict):
    from ..models.model_utils import model_nms_utils

    selected, selected_scores = model_nms_utils.class_agnostic_nms(
        box_scores=pred_dict['pred_scores'], box_preds=pred_dict['pred_boxes'],
        nms_config=EasyDict(
            NMS_TYPE='nms_gpu',
            NMS_THRESH=0,
            NMS_PRE_MAXSIZE=4096,
            NMS_POST_MAXSIZE=500
        ),
        score_thresh=None
    )

    final_scores = selected_scores
    final_labels = pred_dict['pred_labels'][selected]
    final_boxes = pred_dict['pred_boxes'][selected]
    return final_boxes, final_scores, final_labels


def concatenate_preds_from_disk(path):
    """
    Args:
        [
            [{
                'name',
                'score',
                'boxes_lidar',
                'pred_labels',
                'frame_id'
             },
             ...],
            ...,
        ]
    Returns:

    """
    import pickle
    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path

    all_preds = [pickle.load(open(p, 'rb')) for p in Path(path).glob("*.pkl")]
    assert all_preds and np.equal([len(pd) for pd in all_preds], len(all_preds[0])).all()

    pred_dicts = []
    for preds in tqdm(iterable=zip(*all_preds), desc='combing all preds from disk'):
        preds = pd.DataFrame(list(preds)).to_dict(orient="list")  # list of dicts to dict of lists
        preds['name'] = np.concatenate(preds['name'], axis=0)
        preds['score'] = np.concatenate(preds['score'], axis=0)
        preds['pred_labels'] = np.concatenate(preds['pred_labels'], axis=0)
        preds['boxes_lidar'] = np.concatenate([boxes[:, :7] for boxes in preds['boxes_lidar']], axis=0)
        assert len(preds['frame_id']) == preds['frame_id'].count(preds['frame_id'][0])
        pred_dicts.append(preds)

    return pred_dicts
