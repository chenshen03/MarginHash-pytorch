from .ArcFace import ArcMarginProduct
from .MultiMarginProduct import MultiMarginProduct
from .CosFace import CosineMarginProduct
from .SphereFace import SphereProduct
from .InnerProduct import InnerProduct
from .Identity import Identity


def create(name, feat_dim, num_classes, scale):
    if name == 'ArcFace':
        margin = ArcMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'MultiMargin':
        margin = MultiMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'CosFace':
        margin = CosineMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'SphereFace':
        margin = SphereProduct(feat_dim, num_classes)
    elif name == 'Softmax':
        margin = InnerProduct(feat_dim, num_classes)
    elif name == 'Identity':
        margin = Identity()
    else:
        raise ValueError(f'margin {name} is not available!')
    return margin
