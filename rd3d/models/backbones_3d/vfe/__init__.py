from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .dynamic_mean_vfe_v2 import DynamicMeanVFEV2

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'DynamicMeanVFEV2': DynamicMeanVFEV2,
}
