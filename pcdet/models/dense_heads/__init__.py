from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle, AnchorHeadSingle_Aux
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead, CenterHead_Aux_3C
from .anchor_head_semi import AnchorHeadSemi

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'AnchorHeadSemi': AnchorHeadSemi,
    'CenterHead_Aux_3C': CenterHead_Aux_3C,
    'AnchorHeadSingle_Aux': AnchorHeadSingle_Aux,

}
