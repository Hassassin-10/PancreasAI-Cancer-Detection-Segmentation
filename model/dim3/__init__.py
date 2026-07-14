# Robust initialization of dim3 models zoo to ignore missing dependencies

try:
    from .vnet import VNet
except Exception:
    pass

try:
    from .unet import UNet
except Exception:
    pass

try:
    from .unetpp import UNetPlusPlus
except Exception:
    pass

try:
    from .attention_unet import AttentionUNet
except Exception:
    pass

try:
    from .unetr import UNETR
except Exception:
    pass

try:
    from .vtunet import VTUNet
except Exception:
    pass

try:
    from .medformer import MedFormer
except Exception:
    pass

try:
    from .swin_unetr import SwinUNETR
except Exception:
    pass

try:
    from .nnformer import nnFormer
except Exception:
    pass
