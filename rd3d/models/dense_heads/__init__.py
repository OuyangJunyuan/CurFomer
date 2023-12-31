from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_vote import PointHeadVote
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .point_head_IASSD import IASSD_Head
from .point_head_vote_plus import PointHeadVotePlus
from .point_head_vote_pp import PointHeadVotePlusPlus
from .voxelnext_head import VoxelNeXtHead
from .point_segment_head import PointSegmentor

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadVote': PointHeadVote,
    'PointHeadVotePlus': PointHeadVotePlus,
    'PointHeadVote++': PointHeadVotePlusPlus,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'IASSD_Head': IASSD_Head,
    'VoxelNeXtHead': VoxelNeXtHead,
    'PointSegHead': PointSegmentor
}
