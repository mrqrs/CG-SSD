from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .center_points import CenterPoints
from .semi_second import SemiSECOND, SemiSECONDIoU
from .cornernet3d import CornerNet3D, CornerNet3DRCNN
from .corner_rcnn import CornerRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'CenterPoints': CenterPoints,
    'SemiSECOND': SemiSECOND,
    'SemiSECONDIoU': SemiSECONDIoU,
    'CornerNet3D': CornerNet3D,
    'CornerNet3DRCNN': CornerNet3DRCNN,
    'CornerRCNN': CornerRCNN
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
