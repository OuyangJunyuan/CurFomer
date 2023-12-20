from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .point_3dssd import Point3DSSD
from .IASSD import IASSD
from .point_vote import PointVote
from .voxelnext import VoxelNeXt
from .voxformer import VoxFormer

__all__ = {
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    '3DSSD': Point3DSSD,
    'IASSD': IASSD,
    'PointVote': PointVote,
    'VoxelNeXt': VoxelNeXt,
    'VoxFormer': VoxFormer,
}


def build_detector(model_cfg, dataset, class_names=None):
    if class_names is None:
        class_names = model_cfg.get('CLASS_NAMES', None)
    if class_names is None:
        class_names = dataset.class_names
    if class_names is None:
        raise NotImplementedError("miss class_names")

    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=len(class_names), dataset=dataset
    )
    return model
