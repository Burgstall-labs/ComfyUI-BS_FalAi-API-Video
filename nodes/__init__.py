from .node_t2v import FalT2VNode
from .node_i2v import FalI2VNode
from .node_omni import FalAPIOmniProNode
from .node_lipsync import FalLipSyncNode
from .node_omni_v2 import FalOmniProV2Node

NODE_CLASS_MAPPINGS = {
    "FalT2V": FalT2VNode,
    "FalI2V": FalI2VNode,
    "FalOmniPro": FalAPIOmniProNode,
    "FalLipSync": FalLipSyncNode,
    "FalOmniProV2": FalOmniProV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalT2V": "Fal T2V",
    "FalI2V": "Fal I2V",
    "FalOmniPro": "Fal Omni Pro",
    "FalLipSync": "Fal Lip Sync",
    "FalOmniProV2": "Fal Omni Pro v2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
