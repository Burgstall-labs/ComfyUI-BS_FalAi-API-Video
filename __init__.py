from .nodes.node_t2v import FalAPIVideoGeneratorT2V
from .nodes.node_i2v import FalAPIVideoGeneratorI2V
from .nodes.node_omni import FalAPIOmniProNode
from .nodes.node_lipsync import FalAILipSyncNode
from .nodes.node_omni_v2 import FalOmniProV2Node

NODE_CLASS_MAPPINGS = {
    "FalT2V": FalAPIVideoGeneratorT2V,
    "FalI2V": FalAPIVideoGeneratorI2V,
    "FalOmniPro": FalAPIOmniProNode,
    "FalLipSync": FalAILipSyncNode,
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
