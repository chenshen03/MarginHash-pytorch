from .ArcFace import ArcMarginProduct
from .MultiMarginProduct import MultiMarginProduct
from .CosFace import CosineMarginProduct
from .SphereFace import SphereProduct
from .InnerProduct import InnerProduct
from .Identity import Identity
from .AirFace import AirFace


def create(name, feat_dim, num_classes, scale):
    if name == 'ArcFace':
        classifier = ArcMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'MultiMargin':
        classifier = MultiMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'CosFace':
        classifier = CosineMarginProduct(feat_dim, num_classes, s=scale)
    elif name == 'AirFace':
        classifier = AirFace(feat_dim, num_classes, s=scale)
    elif name == 'SphereFace':
        classifier = SphereProduct(feat_dim, num_classes)
    elif name == 'Softmax':
        classifier = InnerProduct(feat_dim, num_classes)
    elif name == 'Identity':
        classifier = Identity()
    else:
        raise ValueError(f'classifier {name} is not available!')
    return classifier
